"""Home & About."""

from flask import Blueprint, render_template

from config import (
    COURSE_INFO,
    PROJECT_DESCRIPTION,
    PROJECT_SUBTITLE,
    PROJECT_TITLE,
    TEAM_MEMBERS,
)

bp = Blueprint("main", __name__)


@bp.route("/")
def home():
    return render_template(
        "home.html",
        title=PROJECT_TITLE,
        subtitle=PROJECT_SUBTITLE,
        description=PROJECT_DESCRIPTION,
        team=TEAM_MEMBERS,
        course=COURSE_INFO,
    )


@bp.route("/about")
def about():
    return render_template(
        "about.html",
        title=PROJECT_TITLE,
        subtitle=PROJECT_SUBTITLE,
        description=PROJECT_DESCRIPTION,
        course=COURSE_INFO,
        team=TEAM_MEMBERS,
    )
