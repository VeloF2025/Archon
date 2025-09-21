class SagaOrchestrator:
    def __init__(self):
        self.steps = []

    def add_step(self, action, compensation):
        self.steps.append((action, compensation))

    async def execute(self):
        for action, compensation in self.steps:
            try:
                await action()
            except Exception as e:
                await self._compensate(compensation)
                raise

    async def _compensate(self, compensation):
        await compensation()