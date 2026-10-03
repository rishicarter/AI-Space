
from pydantic import BaseModel, Field, HttpUrl


class Page(BaseModel):
    """A page discovered on the website."""

    url: HttpUrl
    title: str
    description: str = ""


class Feature(BaseModel):
    """A feature or capability available on a page."""

    name: str
    description: str
    page_url: HttpUrl
    user_role: str = "Unknown"


class Workflow(BaseModel):
    """A sequence of user actions and system responses."""

    name: str
    description: str
    actor: str
    starting_page: HttpUrl
    steps: list[str] = Field(default_factory=list)
    outcome: str = ""


class WebsiteReport(BaseModel):
    """Structured report of the discovered website."""

    website_url: HttpUrl
    website_name: str = ""
    pages: list[Page] = Field(default_factory=list)
    features: list[Feature] = Field(default_factory=list)
    workflows: list[Workflow] = Field(default_factory=list)