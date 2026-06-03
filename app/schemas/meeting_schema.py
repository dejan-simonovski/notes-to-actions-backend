from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, HttpUrl

class TaskPriority(str, Enum):
    """
    Eisenhower Matrix priority levels.
    """
    urgent_important = "urgent_important"
    important_not_urgent = "important_not_urgent"
    urgent_not_important = "urgent_not_important"
    low_priority = "low_priority"

class TaskStatus(str, Enum):
    """
    Status values for action items.
    """
    to_do = "to_do"
    in_progress = "in_progress"
    done = "done"

class ActionItem(BaseModel):
    """
    Schema representing an action item extracted from meeting notes.
    """
    description: str = Field(
        ..., 
        description="A clear and actionable description of the task."
    )
    assignee_name: str = Field(
        ..., 
        description="The name of the person assigned to this task, or 'Unassigned'."
    )
    priority: TaskPriority = Field(
        ..., 
        description="Eisenhower Matrix priority classification."
    )
    status: TaskStatus = Field(
        default=TaskStatus.to_do, 
        description="The current status of the action item."
    )

class AnalyzeRequest(BaseModel):
    """
    Schema for analyzing a meeting transcript from a URL.
    """
    url: Optional[HttpUrl] = Field(
        default=None, 
        description="The Google Doc or web export URL containing the meeting notes transcript."
    )
    title: str = Field(
        ..., 
        description="Title of the meeting."
    )
    date: str = Field(
        ..., 
        description="The date the meeting took place."
    )

class AnalyzeResponse(BaseModel):
    """
    Schema representing the complete analysis result of a meeting transcript.
    """
    title: str = Field(
        ..., 
        description="Title of the meeting."
    )
    summary: str = Field(
        ..., 
        description="A high-level executive summary of the meeting highlights, decisions, and outcomes."
    )
    action_items: List[ActionItem] = Field(
        ..., 
        description="List of action items extracted and prioritized from the meeting notes."
    )
    key_topics: List[str] = Field(
        ...,
        description="List of key topics discussed in the meeting."
    )

class ChatMessage(BaseModel):
    """
    Schema for a single message in the conversation history.
    """
    role: str = Field(..., description="Role of the sender (e.g. 'user' or 'assistant').")
    content: str = Field(..., description="The content of the message.")

class ChatRequest(BaseModel):
    """
    Schema for asking questions about a meeting transcript with conversation history.
    """
    transcript: str = Field(
        ..., 
        min_length=10, 
        description="The full raw text transcript of the meeting."
    )
    question: str = Field(
        ..., 
        min_length=1, 
        description="The specific question the user has about the meeting notes."
    )
    history: Optional[List[ChatMessage]] = Field(
        default=[], 
        description="Previous messages in the conversation to provide context."
    )

class ChatResponse(BaseModel):
    """
    Schema representing the AI's response to a transcript query.
    """
    answer: str = Field(
        ..., 
        description="The direct answer to the user's question, sourced strictly from the transcript."
    )
