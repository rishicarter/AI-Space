from models import Page, Feature, Workflow, WebsiteReport
from pydantic import HttpUrl


def test_website_report():
    report = WebsiteReport(
        website_url="https://dev-task.srivyn.in",
        website_name="Samskar Task Manager",
        pages=[
            Page(
                url="https://dev-task.srivyn.in/login",
                title="Login Page",
            )
        ],
    )

    assert report.website_name == "Samskar Task Manager"
    assert len(report.pages) == 1
    assert isinstance(report.pages[0].url, HttpUrl)