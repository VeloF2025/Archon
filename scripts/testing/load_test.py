import locust

class ArchonLoadTest(locust.HttpUser):
    @locust.task
    def health_check(self):
        self.client.get("/health")