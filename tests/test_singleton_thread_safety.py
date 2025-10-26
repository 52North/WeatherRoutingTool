import threading
from WeatherRoutingTool.algorithms.genetic.patcher import GreatCircleRoutePatcherSingleton


def test_singleton_thread_safety():
    """Test that SingletonBase is thread-safe and only creates one instance across multiple threads."""
    instances = []
    num_threads = 10
    barrier = threading.Barrier(num_threads)

    def create_instance():
        # Synchronize all threads to start at the same time
        barrier.wait()
        instance = GreatCircleRoutePatcherSingleton(dist=100_000.0)
        instances.append(id(instance))

    # Create multiple threads that try to instantiate the singleton simultaneously
    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(target=create_instance)
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Verify that all threads got the same instance
    assert len(instances) == num_threads, "Not all threads completed"
    assert len(set(instances)) == 1, f"Multiple instances created: {set(instances)}"
    print(f"✓ All {num_threads} threads received the same singleton instance (ID: {instances[0]})")


if __name__ == "__main__":
    test_singleton_thread_safety()
    print("Thread-safety test passed!")
