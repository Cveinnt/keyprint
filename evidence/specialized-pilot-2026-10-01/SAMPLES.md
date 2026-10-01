# All specialized pilot outputs

Original text, including formatting and casing failures. No rewriting.

## python_release

Return only Python code, no Markdown. Define can_release(approved, tests_passed, blocked). All inputs are booleans. Return True exactly when approval was received AND tests passed AND release is not blocked. Otherwise return False. Do not import anything or call any functions.

### Key 1 · ordinary · pass

````text
def can_release(approved, tests_passed, blocked):
    return approved and tests_passed and not blocked
````

### Key 1 · marked · pass

````text
def can_release(approved, tests_passed, blocked):
    return approved and tests_passed and not blocked
````

### Key 2 · marked · pass

````text
def can_release(approved, tests_passed, blocked):
    return approved and tests_passed and not blocked
````

### Key 2 · ordinary · pass

````text
def can_release(approved, tests_passed, blocked):
    return approved and tests_passed and not blocked
````

## python_shipping

Return only Python code, no Markdown. Define shipping_fee_cents(total_cents, expedited). total_cents is a nonnegative integer; expedited is boolean. Expedited shipping always costs 1200 cents, even on large orders. Otherwise shipping is free if total_cents is at least 5000, and costs 500 cents below 5000. Return the integer fee. Do not import anything or call any functions.

### Key 1 · marked · not passed

````text
```python
def shipping_fee_cents(total_cents, expedited):
    if expedited:
        return 1200
    elif total_cents >= 5000:
        return 0
    else:
        return 500
```
````

### Key 1 · ordinary · pass

````text
def shipping_fee_cents(total_cents, expedited):
    if expedited:
        return 1200
    if total_cents >= 5000:
        return 0
    return 500
````

### Key 2 · ordinary · pass

````text
def shipping_fee_cents(total_cents, expedited):
    if expedited:
        return 1200
    if total_cents >= 5000:
        return 0
    return 500
````

### Key 2 · marked · pass

````text
def shipping_fee_cents(total_cents, expedited):
    if expedited:
        return 1200
    if total_cents >= 5000:
        return 0
    return 500
````

## sql_approval

Return only one SQLite SELECT query, no Markdown. Table orders(id INTEGER, approved INTEGER, blocked INTEGER). Return id for orders with approved exactly 1 and blocked exactly 0, sorted by id ascending. NULL means unknown and must not be treated as approval or an unblocked state. Do not modify the database.

### Key 1 · ordinary · pass

````text
SELECT id FROM orders WHERE approved = 1 AND blocked = 0 ORDER BY id ASC;
````

### Key 1 · marked · pass

````text
SELECT id FROM orders WHERE approved = 1 AND blocked = 0 ORDER BY id ASC;
````

### Key 2 · marked · pass

````text
SELECT id FROM orders WHERE approved = 1 AND blocked = 0 ORDER BY id ASC;
````

### Key 2 · ordinary · pass

````text
SELECT id FROM orders WHERE approved = 1 AND blocked = 0 ORDER BY id ASC;
````

## sql_inventory

Return only one SQLite SELECT query, no Markdown. Table stock(warehouse TEXT, on_hand INTEGER, reserved INTEGER). All counts are nonnegative and non-NULL. Return warehouse and SUM(on_hand - reserved) AS available, grouped by warehouse. Include only warehouses whose available total is strictly greater than 0. Sort by warehouse ascending. Do not filter individual rows before summing or modify the database.

### Key 1 · marked · pass

````text
SELECT warehouse, SUM(on_hand - reserved) AS available FROM stock GROUP BY warehouse HAVING SUM(on_hand - reserved) > 0 ORDER BY warehouse;
````

### Key 1 · ordinary · pass

````text
SELECT warehouse, SUM(on_hand - reserved) AS available FROM stock GROUP BY warehouse HAVING available > 0 ORDER BY warehouse;
````

### Key 2 · ordinary · pass

````text
SELECT warehouse, SUM(on_hand - reserved) AS available 
FROM stock 
GROUP BY warehouse 
HAVING SUM(on_hand - reserved) > 0 
ORDER BY warehouse;
````

### Key 2 · marked · pass

````text
SELECT warehouse, SUM(on_hand - reserved) AS available FROM stock GROUP BY warehouse HAVING SUM(on_hand - reserved) > 0 ORDER BY warehouse;
````

## json_release

Return only a JSON object with exactly these keys: version, approver, deadline_utc, approved, tests_passed, blocked. Source: version is 3.2.1. Maya is the approver. Deadline is 2026-10-08T09:30:00Z. Approval is still pending. Tests passed. Release remains blocked. Use JSON booleans for the three status fields; do not infer approval from passed tests.

### Key 1 · ordinary · pass

````text
{
  "version": "3.2.1",
  "approver": "Maya",
  "deadline_utc": "2026-10-08T09:30:00Z",
  "approved": false,
  "tests_passed": true,
  "blocked": true
}
````

### Key 1 · marked · pass

````text
{
  "version": "3.2.1",
  "approver": "Maya",
  "deadline_utc": "2026-10-08T09:30:00Z",
  "approved": false,
  "tests_passed": true,
  "blocked": true
}
````

### Key 2 · marked · pass

````text
{
  "version": "3.2.1",
  "approver": "Maya",
  "deadline_utc": "2026-10-08T09:30:00Z",
  "approved": false,
  "tests_passed": true,
  "blocked": true
}
````

### Key 2 · ordinary · pass

````text
{
  "version": "3.2.1",
  "approver": "Maya",
  "deadline_utc": "2026-10-08T09:30:00Z",
  "approved": false,
  "tests_passed": true,
  "blocked": true
}
````

## json_french

Renvoyez uniquement un objet JSON avec exactement ces clés : client, quantite, expedie, jour_expedition_prevu, livraison_confirmee. Source : le client est Élodie Martin ; la commande contient 6 claviers ; elle n'est pas encore expédiée ; l'expédition est prévue vendredi ; aucune date de livraison n'est confirmée. quantite doit être un nombre entier ; expedie et livraison_confirmee des booléens JSON. Conservez le nom exact et le mot français vendredi. Ne confondez pas expédition et livraison.

### Key 1 · marked · pass

````text
{
  "client": "Élodie Martin",
  "quantite": 6,
  "expedie": false,
  "jour_expedition_prevu": "vendredi",
  "livraison_confirmee": false
}
````

### Key 1 · ordinary · pass

````text
{
  "client": "Élodie Martin",
  "quantite": 6,
  "expedie": false,
  "jour_expedition_prevu": "vendredi",
  "livraison_confirmee": false
}
````

### Key 2 · ordinary · pass

````text
{"client": "Élodie Martin", "quantite": 6, "expedie": false, "jour_expedition_prevu": "vendredi", "livraison_confirmee": false}
````

### Key 2 · marked · pass

````text
{
  "client": "Élodie Martin",
  "quantite": 6,
  "expedie": false,
  "jour_expedition_prevu": "vendredi",
  "livraison_confirmee": false
}
````

## csv_timezone

Return only CSV, no Markdown, exactly the header system,start,end,timezone,backup_created,restore_tested then one data row. Source: Kumo maintenance is planned from 13:00 to 13:20, timezone Asia/Tokyo. Backup has been created; restoration has not been tested. Preserve timezone exactly. Represent the two flags as lowercase true or false. Add no date or weekday.

### Key 1 · ordinary · not passed

````text
system,start,end,timezone,backup_created,restore_tested
kumo,13:00,13:20,Asia/Tokyo,true,false
````

### Key 1 · marked · not passed

````text
system,start,end,timezone,backup_created,restore_tested
kumo,13:00,13:20,Asia/Tokyo,true,false
````

### Key 2 · marked · not passed

````text
system,start,end,timezone,backup_created,restore_tested
kumo,13:00,13:20,asia/tokyo,true,false
````

### Key 2 · ordinary · not passed

````text
system,start,end,timezone,backup_created,restore_tested
kumo,13:00,13:20,Asia/Tokyo,true,false
````

## csv_attribution

Return only CSV, no Markdown, with header name,role,status and exactly two rows in the source order. Source: Maya Chen is the approver and approval is pending. Ravi Shah is the recipient and delivery has not happened. Use exact status strings pending and not_delivered, respectively. A recipient is not an approver.

### Key 1 · marked · pass

````text
"name","role","status"
"Maya Chen","approver","pending"
"Ravi Shah","recipient","not_delivered"
````

### Key 1 · ordinary · pass

````text
name,role,status
Maya Chen,approver,pending
Ravi Shah,recipient,not_delivered
````

### Key 2 · ordinary · pass

````text
"name","role","status"
"Maya Chen","approver","pending"
"Ravi Shah","recipient","not_delivered"
````

### Key 2 · marked · pass

````text
name,role,status
Maya Chen,approver,pending
Ravi Shah,recipient,not_delivered
````
