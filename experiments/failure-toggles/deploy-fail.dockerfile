# EXPERIMENT FAILURE FILE
#
# Append this line to the Dockerfile to cause a deployment (Docker build) failure.
# This is used on the experiment/deploy-fail branch.

RUN echo "EXPERIMENT: Intentional Docker build failure" && exit 1
