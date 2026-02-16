"""
API Client Example - Test the Network IDS API
"""

import requests
import numpy as np
import time
import json


class IDSAPIClient:
    """Client for Network IDS API"""

    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def check_health(self):
        """Check API health"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()

    def get_status(self):
        """Get model status"""
        response = requests.get(f"{self.base_url}/status")
        return response.json()

    def get_classes(self):
        """Get attack classes"""
        response = requests.get(f"{self.base_url}/classes")
        return response.json()

    def predict(self, features):
        """Single prediction"""
        response = requests.post(
            f"{self.base_url}/predict", json={"features": features}
        )
        return response.json()

    def predict_batch(self, features_batch):
        """Batch prediction"""
        response = requests.post(
            f"{self.base_url}/predict/batch", json={"traffic_batch": features_batch}
        )
        return response.json()


def test_api():
    """Test all API endpoints"""

    print("=" * 70)
    print("TESTING NETWORK IDS API")
    print("=" * 70)

    client = IDSAPIClient()

    print("\n1. Health Check")
    print("-" * 50)
    try:
        health = client.check_health()
        print(json.dumps(health, indent=2))
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the API server is running!")
        return

    print("-" * 50)
    status = client.get_status()
    print(json.dumps(status, indent=2))

    num_features = status.get("input_features", 50)

    print("\n3. Attack Classes")
    print("-" * 50)
    classes = client.get_classes()
    print(f"Number of classes: {classes['num_classes']}")
    for i, class_name in enumerate(classes["classes"]):
        print(f"  {i}: {class_name}")

    print("\n4. Single Prediction - Simulated Benign Traffic")
    print("-" * 50)
    benign_features = np.random.uniform(0, 1, num_features).tolist()

    start_time = time.time()
    result = client.predict(benign_features)
    elapsed = (time.time() - start_time) * 1000

    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Action: {result['action']}")
    print(f"API Latency: {elapsed:.2f} ms")
    print(f"Server Processing: {result['processing_time_ms']:.2f} ms")

    print("\n5. Single Prediction - Simulated Attack Traffic")
    print("-" * 50)
    attack_features = np.random.uniform(0.7, 1.0, num_features).tolist()

    result = client.predict(attack_features)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Action: {result['action']}")

    print("\nTop 3 Probabilities:")
    sorted_probs = sorted(
        result["all_probabilities"].items(), key=lambda x: x[1], reverse=True
    )
    for class_name, prob in sorted_probs[:3]:
        print(f"  {class_name}: {prob:.4f}")

    print("\n6. Batch Prediction - 10 Samples")
    print("-" * 50)
    batch = [np.random.uniform(0, 1, num_features).tolist() for _ in range(10)]

    start_time = time.time()
    batch_result = client.predict_batch(batch)
    elapsed = (time.time() - start_time) * 1000

    print(f"Total samples: {batch_result['total_samples']}")
    print(f"API Latency: {elapsed:.2f} ms")
    print(
        f"Throughput: {batch_result['total_samples'] / (elapsed / 1000):.2f} samples/sec"
    )

    print("\nPredictions:")
    for pred in batch_result["predictions"][:5]:  # Show first 5
        print(
            f"  Sample {pred['sample_id']}: {pred['prediction']} "
            f"(confidence: {pred['confidence']:.4f}, action: {pred['action']})"
        )
    print(f"  ... and {len(batch_result['predictions']) - 5} more")

    print("\n7. Performance Test - 100 Predictions")
    print("-" * 50)

    num_requests = 100
    latencies = []

    for i in range(num_requests):
        features = np.random.uniform(0, 1, num_features).tolist()
        start = time.time()
        client.predict(features)
        latencies.append((time.time() - start) * 1000)

    print(f"Total requests: {num_requests}")
    print(f"Average latency: {np.mean(latencies):.2f} ms")
    print(f"Min latency: {np.min(latencies):.2f} ms")
    print(f"Max latency: {np.max(latencies):.2f} ms")
    print(f"95th percentile: {np.percentile(latencies, 95):.2f} ms")
    print(f"Throughput: {num_requests / (sum(latencies) / 1000):.2f} req/sec")

    print("\n" + "=" * 70)
    print("API TESTING COMPLETE")
    print("=" * 70)


def simulate_real_time_monitoring():
    """Simulate real-time network monitoring"""

    print("\n" + "=" * 70)
    print("SIMULATING REAL-TIME NETWORK MONITORING")
    print("=" * 70)

    client = IDSAPIClient()

    status = client.get_status()
    num_features = status.get("input_features", 50)

    print("\nMonitoring network traffic (press Ctrl+C to stop)...")
    print("-" * 50)

    attack_count = 0
    benign_count = 0

    try:
        for i in range(50):  # Simulate 50 network flows
            # Simulate network traffic features
            # 80% chance of benign, 20% chance of attack
            if np.random.random() < 0.8:
                features = np.random.uniform(0, 0.5, num_features).tolist()
            else:
                features = np.random.uniform(0.6, 1.0, num_features).tolist()

            result = client.predict(features)

            timestamp = time.strftime("%H:%M:%S")

            if result["action"] == "BLOCK":
                attack_count += 1
                print(
                    f"[{timestamp}] ⚠️  ATTACK DETECTED: {result['prediction']} "
                    f"(confidence: {result['confidence']:.2%})"
                )
            else:
                benign_count += 1
                print(
                    f"[{timestamp}] ✓  Benign traffic "
                    f"(confidence: {result['confidence']:.2%})"
                )

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

    print("\n" + "-" * 50)
    print("Summary:")
    print(f"  Benign traffic: {benign_count}")
    print(f"  Attacks detected: {attack_count}")
    print(f"  Total flows: {benign_count + attack_count}")
    print("=" * 70)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "monitor":
        simulate_real_time_monitoring()
    else:
        test_api()

        print("\n" + "=" * 70)
        response = input("\nRun real-time monitoring simulation? (y/n): ")
        if response.lower() == "y":
            simulate_real_time_monitoring()
