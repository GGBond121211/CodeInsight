# CodeInsight Step 6 local validation image.
# The application invokes this image by the local immutable tag
# `codeinsight-sandbox:py312-v1`; it never accepts an image name from the model.
FROM python:3.12-slim

RUN useradd --create-home --uid 1000 sandbox
RUN pip install --no-cache-dir pytest==9.0.2
WORKDIR /workspace
USER sandbox
