from datetime import datetime, timezone

from risk_persistence.neo4j import Neo4jClient


class Neo4jFeatureProvider:
    def __init__(self, neo4j: Neo4jClient):
        self.neo4j = neo4j

    async def get_graph_features(
        self,
        customer_id: str,
        device_id: str,
        ip_address: str,
        payment_method_id: str,
        merchant_id: str,
        event_time: datetime,
    ) -> dict[str, int]:
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=timezone.utc)

        event_time_iso = event_time.isoformat()

        query = """
        CALL () {
            MATCH (c:Customer {id: $customer_id})-[r]-()
            WHERE
                (
                    type(r) = 'MADE'
                    AND endNode(r).event_time < datetime($event_time)
                )
                OR (
                    type(r) IN ['USED_DEVICE', 'USED_IP', 'USED_PAYMENT']
                    AND r.first_seen_at < datetime($event_time)
                )
            RETURN count(r) AS customer_degree
        }

        CALL () {
            MATCH (d:Device {id: $device_id})<-[r:USED_DEVICE]-(c:Customer)
            WHERE r.first_seen_at < datetime($event_time)
            RETURN count(DISTINCT c) AS device_customer_count
        }

        CALL () {
            MATCH (ip:IP {address: $ip_address})<-[r:USED_IP]-(c:Customer)
            WHERE r.first_seen_at < datetime($event_time)
            RETURN count(DISTINCT c) AS ip_customer_count
        }

        CALL () {
            MATCH (p:PaymentMethod {id: $payment_method_id})<-[r:USED_PAYMENT]-(c:Customer)
            WHERE r.first_seen_at < datetime($event_time)
            RETURN count(DISTINCT c) AS payment_customer_count
        }

        CALL () {
            MATCH (:Merchant {id: $merchant_id})<-[:AT_MERCHANT]-(t:Transaction)
            WHERE t.event_time < datetime($event_time)
            RETURN count(DISTINCT t) AS merchant_transaction_count
        }

        RETURN
            customer_degree,
            device_customer_count,
            ip_customer_count,
            payment_customer_count,
            merchant_transaction_count
        """

        result = await self.neo4j.execute(
            query,
            {
                "customer_id": customer_id,
                "device_id": device_id,
                "ip_address": ip_address,
                "payment_method_id": payment_method_id,
                "merchant_id": merchant_id,
                "event_time": event_time_iso,
            },
        )

        if not result:
            return {
                "customer_degree": 0,
                "device_customer_count": 0,
                "ip_customer_count": 0,
                "payment_customer_count": 0,
                "merchant_transaction_count": 0,
            }

        row = result[0]

        return {
            "customer_degree": int(row["customer_degree"]),
            "device_customer_count": int(row["device_customer_count"]),
            "ip_customer_count": int(row["ip_customer_count"]),
            "payment_customer_count": int(row["payment_customer_count"]),
            "merchant_transaction_count": int(
                row["merchant_transaction_count"]
            ),
        }