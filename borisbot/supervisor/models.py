from pydantic import BaseModel
from typing import Optional, List
from enum import Enum
from datetime import datetime

class AgentStatus(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    CRASHED = "crashed"
    STARTING = "starting"

class AgentCreate(BaseModel):
    name: str
    parent_id: Optional[str] = None
    autonomy_mode: str = "manual"

class AgentResponse(BaseModel):
    id: str
    name: str
    status: AgentStatus
    parent_id: Optional[str]
    autonomy_mode: str
    created_at: datetime
    last_heartbeat: Optional[datetime] = None
    pid: Optional[int]

class CapabilityCreate(BaseModel):
    capability_type: str
    capability_value: str

class CapabilityResponse(CapabilityCreate):
    id: int
    agent_id: str
