from fastapi import Request, HTTPException
from .access_control import AccessControlManager as AccessControl

class RBACMiddleware:
    def __init__(self, app):
        self.app = app
        self.access_control = AccessControl()

    async def __call__(self, scope, receive, send):
        request = Request(scope)
        user_role = request.headers.get('user-role', 'guest')  # Example; replace with actual auth
        if not self.access_control.check_permission(user_role, request.url.path):
            raise HTTPException(status_code=403, detail="Forbidden")
        await self.app(scope, receive, send)