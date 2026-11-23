from decimal import Decimal

from app.cart_calculator import calculate_cart


def test_cart_calculation_is_deterministic():
    cart = calculate_cart([
        {
            "item_id": 1,
            "name": "برجر",
            "base_price": "95",
            "quantity": 2,
            "modifiers": [{"name": "جبنة", "price": "12"}],
            "notes": "",
        }
    ], Decimal("20"))

    assert cart["subtotal"] == "214.00"
    assert cart["total_amount"] == "234.00"


def test_cart_rejects_zero_quantity():
    try:
        calculate_cart([{
            "item_id": 1,
            "name": "برجر",
            "base_price": "95",
            "quantity": 0,
        }])
    except ValueError:
        return
    raise AssertionError("zero quantity should be rejected")
