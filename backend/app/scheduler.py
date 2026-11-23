"""Durable Database-Backed Asynchronous Job Scheduler for Wasla AI.

Guarantees:
- Persisted in `scheduled_jobs` table (PostgreSQL / SQLite) — survives application restarts.
- Strict idempotency via unique `idempotency_key` (never duplicates reviews or notifications).
- Primary review trigger: exactly 60 minutes (1 hour) after order status -> DELIVERED / FULFILLED.
- Fallback trigger: 90 minutes after order confirmation if delivery is NOT yet confirmed -> arrival check.
- Automatic review suppression for: Cancelled, Failed, Refunded, Unresolved Critical Complaints,
  Opted-out customers, or already-sent reviews.
- Safe retry with exponential backoff + DEAD_LETTER handling + dashboard alerts.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    ChatSession,
    ComplaintTicket,
    FeedbackRequest,
    Order,
    OrderEvent,
    OrderStatus,
    Outbox,
    ScheduledJob,
)
from app.notifications import NotificationService

logger = logging.getLogger("wasla.scheduler")

DELIVERED_STATUSES = {OrderStatus.DELIVERED.value, OrderStatus.FULFILLED.value}
SUPPRESSED_ORDER_STATUSES = {OrderStatus.CANCELLED.value, OrderStatus.FAILED.value}


class DurableJobScheduler:
    """Restart-safe, idempotent DB-backed job scheduler."""

    def __init__(
        self,
        notification_service: NotificationService,
        review_delay_minutes: int = 60,
        arrival_check_delay_minutes: int = 90,
        lease_seconds: int = 120,
        worker_id: str | None = None,
    ):
        self.notification_service = notification_service
        self.review_delay_minutes = review_delay_minutes
        self.arrival_check_delay_minutes = arrival_check_delay_minutes
        self.lease_seconds = lease_seconds
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"

    def recover_stale_jobs_on_startup(self, db: Session, now: datetime | None = None) -> int:
        """Reset any jobs left in RUNNING state after a crash/restart or expired lease."""
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        stale = db.scalars(select(ScheduledJob).where(ScheduledJob.status == "RUNNING")).all()
        recovered = 0
        for job in stale:
            lu = job.locked_until
            if lu is not None and lu.tzinfo is None:
                lu = lu.replace(tzinfo=timezone.utc)
            if lu is None or lu <= now:
                job.status = "PENDING"
                job.locked_by = None
                job.locked_until = None
                recovered += 1
        if recovered:
            db.commit()
        return recovered

    def schedule_job(
        self,
        db: Session,
        *,
        idempotency_key: str,
        job_type: str,
        run_at: datetime,
        order_id: int | None = None,
        customer_phone: str = "",
        payload: dict[str, Any] | None = None,
        max_attempts: int = 5,
    ) -> ScheduledJob:
        existing = db.scalar(select(ScheduledJob).where(ScheduledJob.idempotency_key == idempotency_key))
        if existing is not None:
            return existing
        job = ScheduledJob(
            idempotency_key=idempotency_key,
            job_type=job_type,
            order_id=order_id,
            customer_phone=customer_phone,
            payload=payload or {},
            run_at=run_at,
            status="PENDING",
            attempts=0,
            max_attempts=max_attempts,
        )
        db.add(job)
        db.flush()
        return job

    def schedule_post_confirmation_arrival_check(self, db: Session, order: Order, confirmed_at: datetime | None = None) -> ScheduledJob:
        base = confirmed_at or order.confirmed_at or datetime.now(timezone.utc)
        return self.schedule_job(
            db,
            idempotency_key=f"arrival_check_90m:{order.id}",
            job_type="arrival_check",
            run_at=base + timedelta(minutes=self.arrival_check_delay_minutes),
            order_id=order.id,
            customer_phone=order.customer_phone,
            payload={"delay_minutes": self.arrival_check_delay_minutes},
        )

    def schedule_post_delivery_review(self, db: Session, order: Order, delivered_at: datetime | None = None) -> tuple[FeedbackRequest, ScheduledJob]:
        base = delivered_at or order.delivered_at or datetime.now(timezone.utc)
        due_at = base + timedelta(minutes=self.review_delay_minutes)
        feedback = db.scalar(select(FeedbackRequest).where(FeedbackRequest.order_id == order.id))
        if feedback is None:
            feedback = FeedbackRequest(
                order_id=order.id,
                customer_phone=order.customer_phone,
                token=str(uuid.uuid4()),
                due_at=due_at,
            )
            db.add(feedback)
            db.flush()
            db.add(
                OrderEvent(
                    order_id=order.id,
                    event_type="feedback_scheduled",
                    payload={"delay_minutes": self.review_delay_minutes, "due_at": due_at.isoformat()},
                )
            )
        job = self.schedule_job(
            db,
            idempotency_key=f"review_1h:{order.id}",
            job_type="send_review_request",
            run_at=due_at,
            order_id=order.id,
            customer_phone=order.customer_phone,
            payload={"feedback_token": feedback.token, "delay_minutes": self.review_delay_minutes},
        )
        return feedback, job

    def schedule_cashier_retry(self, db: Session, order: Order, delay_seconds: int = 30) -> ScheduledJob:
        return self.schedule_job(
            db,
            idempotency_key=f"cashier_notify_retry:{order.id}",
            job_type="cashier_order_notification",
            run_at=datetime.now(timezone.utc) + timedelta(seconds=delay_seconds),
            order_id=order.id,
            customer_phone=order.customer_phone,
            payload={"order_id": order.id},
            max_attempts=5,
        )

    def schedule_owner_complaint_retry(self, db: Session, ticket: ComplaintTicket, delay_seconds: int = 30) -> ScheduledJob:
        return self.schedule_job(
            db,
            idempotency_key=f"owner_complaint_retry:{ticket.id}",
            job_type="owner_complaint_notification",
            run_at=datetime.now(timezone.utc) + timedelta(seconds=delay_seconds),
            order_id=ticket.order_id,
            customer_phone=ticket.customer_phone,
            payload={"ticket_id": ticket.id},
            max_attempts=5,
        )

    def check_review_suppression(self, db: Session, order: Order, feedback: FeedbackRequest | None) -> str | None:
        """Return suppression reason if review MUST NOT be sent, else None."""
        status_str = order.status.value if hasattr(order.status, "value") else str(order.status)
        if status_str in SUPPRESSED_ORDER_STATUSES:
            return f"order_status_{status_str.lower()}"
        if status_str not in DELIVERED_STATUSES:
            return "delivery_not_confirmed"
        if getattr(order, "refunded", False):
            return "order_refunded"
        if feedback is not None and feedback.sent_at is not None:
            return "review_already_sent"
        session = db.scalar(select(ChatSession).where(ChatSession.phone_number == order.customer_phone))
        if session is not None and session.opted_out_reviews:
            return "customer_opted_out"
        unresolved_critical = db.scalar(
            select(ComplaintTicket).where(
                ComplaintTicket.order_id == order.id,
                ComplaintTicket.severity.in_(["CRITICAL", "HIGH"]),
                ComplaintTicket.status != "RESOLVED",
            )
        )
        if unresolved_critical is not None:
            return f"unresolved_complaint_ticket_{unresolved_critical.id}"
        return None

    def claim_due_jobs(self, db: Session, now: datetime) -> list[ScheduledJob]:
        """Atomically claim due PENDING/RETRYING/expired-lease jobs with worker lease protection."""
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        lease_until = now + timedelta(seconds=self.lease_seconds)

        # Reclaim expired leases from crashed/sleeping workers first
        running_jobs = db.scalars(select(ScheduledJob).where(ScheduledJob.status == "RUNNING")).all()
        reclaimed_any = False
        for rj in running_jobs:
            lu = rj.locked_until
            if lu is not None and lu.tzinfo is None:
                lu = lu.replace(tzinfo=timezone.utc)
            if lu is not None and lu <= now:
                rj.status = "RETRYING"
                rj.locked_by = None
                rj.locked_until = None
                reclaimed_any = True
        if reclaimed_any:
            db.flush()

        stmt = (
            select(ScheduledJob)
            .where(ScheduledJob.status.in_(["PENDING", "RETRYING"]))
            .order_by(ScheduledJob.run_at.asc())
        )
        dialect_name = db.bind.dialect.name if db.bind is not None else ""
        if dialect_name == "postgresql":
            stmt = stmt.with_for_update(skip_locked=True)

        candidates = db.scalars(stmt).all()
        claimed: list[ScheduledJob] = []
        for job in candidates:
            run_at = job.run_at
            if run_at.tzinfo is None:
                run_at = run_at.replace(tzinfo=timezone.utc)
            if run_at > now:
                continue
            lu = job.locked_until
            if lu is not None and lu.tzinfo is None:
                lu = lu.replace(tzinfo=timezone.utc)
            if job.locked_by and lu is not None and lu > now:
                continue
            job.status = "RUNNING"
            job.locked_by = self.worker_id
            job.locked_until = lease_until
            job.attempts = (job.attempts or 0) + 1
            claimed.append(job)

        if claimed:
            db.commit()
        return claimed

    async def process_due_jobs(self, db: Session, now: datetime | None = None) -> list[dict[str, Any]]:
        """Execute all due/missed jobs cleanly with lease locking and retry/dead-letter handling."""
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        claimed_jobs = self.claim_due_jobs(db, now)
        results: list[dict[str, Any]] = []
        for job in claimed_jobs:
            try:
                outcome = await self._execute_single_job(db, job, now)
                job.locked_by = None
                job.locked_until = None
                results.append({"job_id": job.id, "job_type": job.job_type, "worker_id": self.worker_id, "outcome": outcome})
            except Exception as exc:
                logger.exception("Scheduled job %s (%s) failed: %s", job.id, job.job_type, exc)
                job.last_error = str(exc)
                job.locked_by = None
                job.locked_until = None
                if job.attempts >= job.max_attempts:
                    job.status = "DEAD_LETTER"
                    db.add(
                        Outbox(
                            event_id=str(uuid.uuid4()),
                            event_type="scheduler_dead_letter",
                            payload={"job_id": job.id, "job_type": job.job_type, "order_id": job.order_id, "error": str(exc)},
                        )
                    )
                else:
                    job.status = "RETRYING"
                    job.run_at = now + timedelta(seconds=30 * (2 ** (job.attempts - 1)))
                results.append({"job_id": job.id, "job_type": job.job_type, "error": str(exc), "status": job.status})

        db.commit()
        return results

    async def _execute_single_job(self, db: Session, job: ScheduledJob, now: datetime) -> dict[str, Any]:
        if job.job_type == "owner_complaint_notification":
            ticket_id = (job.payload or {}).get("ticket_id")
            ticket = db.get(ComplaintTicket, ticket_id) if ticket_id else None
            if ticket is None or ticket.status == "RESOLVED" or ticket.owner_notified:
                job.status = "SUPPRESSED"
                job.last_error = "ticket_missing_resolved_or_already_notified"
                job.completed_at = now
                return {"status": "SUPPRESSED", "reason": job.last_error}
            res = await self.notification_service.escalate_complaint_to_owner(db, ticket)
            if res.get("status") == "FAILED":
                raise RuntimeError(res.get("error") or "Owner Telegram escalation failed")
            job.status = "COMPLETED"
            job.completed_at = now
            return {"status": "COMPLETED", "delivery": res}

        order = db.get(Order, job.order_id) if job.order_id else None
        if order is None:
            job.status = "SUPPRESSED"
            job.last_error = "order_not_found"
            job.completed_at = now
            return {"status": "SUPPRESSED", "reason": "order_not_found"}

        if job.job_type == "send_review_request":
            feedback = db.scalar(select(FeedbackRequest).where(FeedbackRequest.order_id == order.id))
            suppression = self.check_review_suppression(db, order, feedback)
            if suppression:
                if feedback is not None:
                    feedback.suppressed_reason = suppression
                job.status = "SUPPRESSED"
                job.last_error = suppression
                job.completed_at = now
                db.add(OrderEvent(order_id=order.id, event_type="review_suppressed", payload={"reason": suppression}))
                return {"status": "SUPPRESSED", "reason": suppression}

            assert feedback is not None
            res = await self.notification_service.send_customer_review_request(db, order, feedback, now=now)
            if res.get("policy_blocked"):
                # Retain blocked job transparently without spending money
                job.status = "BLOCKED_POLICY"
                job.last_error = res.get("reason")
                db.add(OrderEvent(order_id=order.id, event_type="review_blocked_policy", payload={"reason": res.get("reason")}))
                return {"status": "BLOCKED_POLICY", "reason": res.get("reason")}
            if res.get("status") == "FAILED":
                raise RuntimeError(res.get("error", "Review send failed"))

            job.status = "COMPLETED"
            job.completed_at = now
            db.add(
                Outbox(
                    event_id=str(uuid.uuid4()),
                    event_type="feedback_request_due",
                    payload={"order_id": order.id, "customer_phone": order.customer_phone, "feedback_token": feedback.token},
                )
            )
            db.add(OrderEvent(order_id=order.id, event_type="feedback_request_sent", payload={"feedback_token": feedback.token}))
            return {"status": "COMPLETED", "delivery": res}

        if job.job_type == "arrival_check":
            status_str = order.status.value if hasattr(order.status, "value") else str(order.status)
            if status_str in DELIVERED_STATUSES or status_str in SUPPRESSED_ORDER_STATUSES:
                job.status = "SUPPRESSED"
                job.last_error = f"already_{status_str.lower()}"
                job.completed_at = now
                return {"status": "SUPPRESSED", "reason": job.last_error}

            res = await self.notification_service.send_arrival_check(db, order, now=now)
            if res.get("policy_blocked"):
                job.status = "BLOCKED_POLICY"
                job.last_error = res.get("reason")
                db.add(OrderEvent(order_id=order.id, event_type="arrival_check_blocked_policy", payload={"reason": res.get("reason")}))
                return {"status": "BLOCKED_POLICY", "reason": res.get("reason")}
            if res.get("status") == "FAILED":
                raise RuntimeError(res.get("error", "Arrival check send failed"))

            job.status = "COMPLETED"
            job.completed_at = now
            db.add(OrderEvent(order_id=order.id, event_type="arrival_check_sent", payload=res))
            return {"status": "COMPLETED", "delivery": res}

        if job.job_type == "cashier_order_notification":
            res = await self.notification_service.notify_cashier_new_order(db, order)
            if res.get("status") == "FAILED":
                raise RuntimeError(res.get("error", "Cashier Telegram delivery failed"))
            job.status = "COMPLETED"
            job.completed_at = now
            return {"status": "COMPLETED", "delivery": res}

        job.status = "SUPPRESSED"
        job.last_error = f"unknown_job_type:{job.job_type}"
        job.completed_at = now
        return {"status": "SUPPRESSED"}
