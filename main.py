"""FastAPI 应用入口。

启动方式：
    uvicorn main:app --reload          # 开发模式
    python main.py                     # 直接运行（读取 Settings.debug）

阶段一仅提供：
    GET  /health                  健康检查
    POST /api/v1/memory/facts     MemoryFact 校验回显端点
"""

import uvicorn
from fastapi import FastAPI

from app.api.routes import router as api_router
from app.core.config import settings


def create_app() -> FastAPI:
    """应用工厂：集中组装中间件、路由与生命周期钩子。

    采用工厂函数而非模块级直建，便于测试中创建隔离的应用实例。

    Returns:
        配置完成的 FastAPI 应用实例。
    """
    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="基于统计优化的长程记忆与自纠错智能体系统",
    )

    application.include_router(api_router)

    @application.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        """健康检查端点。"""
        return {
            "status": "ok",
            "app": settings.app_name,
            "version": settings.app_version,
        }

    return application


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=settings.debug,
    )
