import requests
import time

# =========================================================
# CONFIGURATION
# =========================================================
TB_URL = "https://windforce.thingsnode.cc"
USERNAME = ""
PASSWORD = ""

ROOT_ASSET_ID = "78fda490-e08b-11f0-b68f-8f33a9d74e0c"   # Level 0 asset
TARGET_LEVEL = 3                   # We want Level-4 assets

# =========================================================
# AUTHENTICATION
# =========================================================
def tb_login():
    url = f"{TB_URL}/api/auth/login"
    payload = {"username": USERNAME, "password": PASSWORD}
    r = requests.post(url, json=payload)
    r.raise_for_status()
    token = r.json()["token"]
    return {"X-Authorization": f"Bearer {token}"}

HEADERS = tb_login()
print("✅ Logged into ThingsBoard")

# =========================================================
# RELATION FUNCTIONS
# =========================================================
def get_asset_children(asset_id):
    """Get child ASSET relations"""
    url = f"{TB_URL}/api/relations/info"
    params = {"fromId": asset_id, "fromType": "ASSET"}
    r = requests.get(url, headers=HEADERS, params=params)
    r.raise_for_status()

    children = []
    for rel in r.json():
        if rel["to"]["entityType"] == "ASSET" and rel["type"] == "Contains":
            children.append(rel["to"]["id"])
    return children


def get_devices_from_asset(asset_id):
    """Get DEVICE relations under an asset"""
    url = f"{TB_URL}/api/relations/info"
    params = {"fromId": asset_id, "fromType": "ASSET"}
    r = requests.get(url, headers=HEADERS, params=params)
    r.raise_for_status()

    devices = []
    for rel in r.json():
        if rel["to"]["entityType"] == "DEVICE":
            devices.append(rel["to"]["id"])
    return devices


def get_device_name(device_id):
    """Get device name from device ID"""
    url = f"{TB_URL}/api/device/{device_id}"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    return r.json().get("name", device_id)


# =========================================================
# TRAVERSE TO LEVEL-N ASSETS
# =========================================================
def get_assets_at_level(root_asset_id, level):
    current_assets = [root_asset_id]

    for lvl in range(1, level+1):
        next_assets = []
        for aid in current_assets:
            next_assets.extend(get_asset_children(aid))

        current_assets = next_assets
        print(f"➡ Level {lvl}: {len(current_assets)} assets found")

        if not current_assets:
            break

    return current_assets


# =========================================================
# TELEMETRY WRITER
# =========================================================
def write_device_telemetry(device_id, telemetry):
    url = f"{TB_URL}/api/plugins/telemetry/DEVICE/{device_id}/timeseries/ANY"
    r = requests.post(url, headers=HEADERS, json=telemetry)
    r.raise_for_status()


# =========================================================
# MAIN FLOW
# =========================================================
if __name__ == "__main__":

    # Step 1: Get Level-4 assets
    level3_assets = get_assets_at_level(ROOT_ASSET_ID, TARGET_LEVEL)
    print(f"\n✅ Total Level-4 Assets: {len(level3_assets)}")

    # Step 2: Loop through each Level-4 asset
    for asset_id in level3_assets:
        # Step 3: Get devices under the asset
        devices = get_devices_from_asset(asset_id)
        print(f"\nAsset {asset_id} → {len(devices)} devices")

        # Step 4: Send telemetry to each device
        for device_id in devices:
            device_name = get_device_name(device_id)
            telemetry_data = {
                "predicted_value": 42.0,
                "timestamp": int(time.time() * 1000)
            }

            write_device_telemetry(device_id, telemetry_data)
            print(f"   ✔ Telemetry sent to device {device_name} (ID: {device_id})")

    print("\n🎉 Telemetry successfully written to all Level-4 devices")
