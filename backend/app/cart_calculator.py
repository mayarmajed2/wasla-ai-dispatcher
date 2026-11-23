from decimal import Decimal, ROUND_HALF_UP

MONEY = Decimal("0.01")


def money(value) -> Decimal:
    return Decimal(str(value)).quantize(MONEY, rounding=ROUND_HALF_UP)


def calculate_cart(items: list[dict], delivery_fee: Decimal = Decimal("0")) -> dict:
    subtotal = Decimal("0")
    snapshot = []

    for item in items:
        unit_price = money(item["base_price"])
        modifier_total = sum(
            (money(modifier["price"]) for modifier in item.get("modifiers", [])),
            Decimal("0"),
        )
        quantity = int(item["quantity"])
        if quantity < 1:
            raise ValueError("quantity must be at least 1")
        line_total = money((unit_price + modifier_total) * quantity)
        subtotal += line_total
        snapshot.append({
            "item_id": item["item_id"],
            "name": item["name"],
            "quantity": quantity,
            "unit_price": str(unit_price),
            "modifiers": item.get("modifiers", []),
            "notes": item.get("notes", ""),
            "line_total": str(line_total),
        })

    subtotal = money(subtotal)
    delivery_fee = money(delivery_fee)
    return {
        "items": snapshot,
        "subtotal": str(subtotal),
        "delivery_fee": str(delivery_fee),
        "total_amount": str(money(subtotal + delivery_fee)),
    }
