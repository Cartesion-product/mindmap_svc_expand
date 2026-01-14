"""Celery 应用模块"""
from celery_app.celery_config import celery_app
from celery_app.tasks import generate_mindmap_task
