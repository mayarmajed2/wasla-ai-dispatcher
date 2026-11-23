# Restaurant AI Dispatcher (MOKA)

Commercial AI Intelligence Layer for Egyptian Arabic WhatsApp Restaurant Ordering (Voice-First + Text + Human-in-the-Loop Cashier Directives).

## Project Status

- **Current Phase:** Phase 0–2 (Workspace Inspection, Menu Extraction, Architecture & Roadmap Approval Gate)
- **Implementation Status:** Pending explicit architecture & roadmap approval before AI service coding begins.
- **Existing Backend & Cashier Dashboard:** Developed independently by another team member and not currently present in this workspace. This repository develops the **independent AI intelligence layer** with clean integration contracts and a temporary Mock Backend for isolated testing.

## Repository Structure

```text
.
├── docs/
│   └── RESTAURANT_AI_PROJECT_ROADMAP.pdf   # Comprehensive 28-section commercial roadmap
├── menu/
│   ├── menu.json                           # Full structured MOKA menu (source of truth)
│   ├── categories.json                     # Extracted menu categories
│   ├── products.json                       # Extracted products, prices, sizes, descriptions
│   ├── modifiers.json                      # Extracted extras and modifiers
│   └── extraction_report.md                # Extraction summary & items awaiting confirmation
├── المواصفات الفنية للنظام.pdf               # Existing restaurant technical specification (read-only)
├── WhatsApp Image *.jpeg                   # Source restaurant menu images (read-only)
├── .env.example                            # Environment variable template (no secrets)
├── .gitignore                              # Git ignore rules
└── README.md                               # This file
```

## Key Architectural Principles

1. **Deterministic Business Safety:** The AI layer never invents prices, discounts, delivery fees, or item availability. Authoritative business state and financial calculations live in the Backend.
2. **Egyptian Arabic Voice-First:** Built specifically for Egyptian colloquial Arabic text and WhatsApp voice notes (`audio/ogg`).
3. **Human-in-the-Loop:** Preserves full cashier operational control for unavailable items, kitchen delays, unreachable deliveries, and free-text cashier directives.
4. **Clean Integration Contracts:** Decoupled from the unavailable Backend via well-defined service interfaces (`MenuProvider`, `CartService`, `DeliveryService`, `OrderService`, `HumanHandoff`) and tested against an isolated Mock Backend adapter.

## Next Steps

1. Review `menu/extraction_report.md` and confirm unclear menu details.
2. Review `docs/RESTAURANT_AI_PROJECT_ROADMAP.pdf` and approve architecture, model selection, and audio pipeline options.
3. Begin independent AI layer development (Phase 3+).
