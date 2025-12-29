#!/usr/bin/env bash
#
# Azure infrastructure provisioning script for CI/CD thesis experiment.
#
# Prerequisites:
#   - Azure CLI installed and logged in (az login)
#   - Subscription selected (az account set --subscription <id>)
#
# Usage:
#   # With defaults:
#   ./scripts/azure_setup.sh
#
#   # With custom values:
#   RG=my-rg LOCATION=northeurope ACR_NAME=myacr ./scripts/azure_setup.sh
#
set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration (override via environment variables)
# -----------------------------------------------------------------------------
RG="${RG:-thesis-cicd-rg}"
LOCATION="${LOCATION:-westeurope}"
ACR_NAME="${ACR_NAME:-thesiscicdacr}"
PLAN_NAME="${PLAN_NAME:-thesis-cicd-plan}"
WEBAPP_NAME="${WEBAPP_NAME:-thesis-cicd-app}"
SKU="${SKU:-B1}"

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

check_az_cli() {
    if ! command -v az &> /dev/null; then
        echo "ERROR: Azure CLI (az) not found. Install from https://aka.ms/install-azure-cli"
        exit 1
    fi

    if ! az account show &> /dev/null; then
        echo "ERROR: Not logged in to Azure. Run 'az login' first."
        exit 1
    fi
}

register_providers() {
    local providers=("Microsoft.ContainerRegistry" "Microsoft.Web")

    for provider in "${providers[@]}"; do
        local state
        state=$(az provider show --namespace "$provider" --query "registrationState" -o tsv 2>/dev/null || echo "NotRegistered")

        if [[ "$state" != "Registered" ]]; then
            log "Registering provider: $provider"
            az provider register --namespace "$provider" --wait
        else
            log "Provider already registered: $provider"
        fi
    done
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
log "Starting Azure provisioning..."
log "Configuration:"
log "  Resource Group: $RG"
log "  Location:       $LOCATION"
log "  ACR Name:       $ACR_NAME"
log "  App Plan:       $PLAN_NAME"
log "  Web App:        $WEBAPP_NAME"
log "  SKU:            $SKU"
echo ""

check_az_cli

# 0. Register required resource providers
log "Checking resource providers..."
register_providers

# 1. Create Resource Group
log "Creating Resource Group: $RG"
az group create \
    --name "$RG" \
    --location "$LOCATION" \
    --output none

# 2. Create Azure Container Registry
log "Creating Azure Container Registry: $ACR_NAME"
az acr create \
    --resource-group "$RG" \
    --name "$ACR_NAME" \
    --sku Basic \
    --admin-enabled true \
    --output none

# Get ACR login server
ACR_LOGIN_SERVER=$(az acr show --name "$ACR_NAME" --query loginServer --output tsv)
log "ACR Login Server: $ACR_LOGIN_SERVER"

# 3. Create App Service Plan (Linux)
log "Creating App Service Plan: $PLAN_NAME"
az appservice plan create \
    --resource-group "$RG" \
    --name "$PLAN_NAME" \
    --is-linux \
    --sku "$SKU" \
    --output none

# 4. Create Web App for Containers
log "Creating Web App: $WEBAPP_NAME"
az webapp create \
    --resource-group "$RG" \
    --plan "$PLAN_NAME" \
    --name "$WEBAPP_NAME" \
    --deployment-container-image-name "mcr.microsoft.com/appsvc/staticsite:latest" \
    --output none

# 5. Configure Web App to use ACR
log "Configuring Web App to use ACR..."
ACR_USERNAME=$(az acr credential show --name "$ACR_NAME" --query username --output tsv)
ACR_PASSWORD=$(az acr credential show --name "$ACR_NAME" --query "passwords[0].value" --output tsv)

az webapp config container set \
    --resource-group "$RG" \
    --name "$WEBAPP_NAME" \
    --docker-registry-server-url "https://$ACR_LOGIN_SERVER" \
    --docker-registry-server-user "$ACR_USERNAME" \
    --docker-registry-server-password "$ACR_PASSWORD" \
    --output none

# 6. Enable container logging
log "Enabling container logging..."
az webapp log config \
    --resource-group "$RG" \
    --name "$WEBAPP_NAME" \
    --docker-container-logging filesystem \
    --output none

# 7. Set app settings
log "Configuring app settings..."
az webapp config appsettings set \
    --resource-group "$RG" \
    --name "$WEBAPP_NAME" \
    --settings WEBSITES_PORT=8000 \
    --output none

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
log "===== Provisioning Complete ====="
echo ""
echo "Resources created:"
echo "  Resource Group:  $RG"
echo "  ACR:             $ACR_LOGIN_SERVER"
echo "  App Service:     https://$WEBAPP_NAME.azurewebsites.net"
echo ""
echo "Next steps:"
echo "  1. Create a service principal for GitHub Actions:"
echo ""
echo "     az ad sp create-for-rbac \\"
echo "       --name \"github-actions-thesis\" \\"
echo "       --role contributor \\"
echo "       --scopes /subscriptions/\$(az account show --query id -o tsv)/resourceGroups/$RG \\"
echo "       --sdk-auth"
echo ""
echo "  2. Copy the JSON output and add it as GitHub secret: AZURE_CREDENTIALS"
echo ""
echo "  3. Add these GitHub repository variables:"
echo "     ACR_LOGIN_SERVER = $ACR_LOGIN_SERVER"
echo "     AZURE_WEBAPP_NAME = $WEBAPP_NAME"
echo "     AZURE_RG = $RG"
echo ""
