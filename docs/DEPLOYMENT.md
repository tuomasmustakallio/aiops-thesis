# Azure Deployment Guide

This document describes how to set up Azure infrastructure and GitHub Actions for automated deployment.

## Prerequisites

- Azure CLI installed (`az --version`)
- Azure subscription
- GitHub repository with Actions enabled

## 1. Azure Infrastructure Setup

### Option A: Run the provisioning script

```bash
# Login to Azure
az login

# (Optional) Select subscription
az account set --subscription "<subscription-id>"

# Run with defaults
./scripts/azure_setup.sh

# Or with custom values
RG=my-rg LOCATION=northeurope ACR_NAME=myacr WEBAPP_NAME=myapp ./scripts/azure_setup.sh
```

**Default values:**
| Variable | Default |
|----------|---------|
| RG | thesis-cicd-rg |
| LOCATION | westeurope |
| ACR_NAME | thesiscicdacr |
| PLAN_NAME | thesis-cicd-plan |
| WEBAPP_NAME | thesis-cicd-app |
| SKU | B1 |

### Option B: Manual setup

```bash
# Variables
RG="thesis-cicd-rg"
LOCATION="westeurope"
ACR_NAME="thesiscicdacr"
PLAN_NAME="thesis-cicd-plan"
WEBAPP_NAME="thesis-cicd-app"

# Create Resource Group
az group create --name $RG --location $LOCATION

# Create Azure Container Registry
az acr create --resource-group $RG --name $ACR_NAME --sku Basic --admin-enabled true

# Create App Service Plan (Linux)
az appservice plan create --resource-group $RG --name $PLAN_NAME --is-linux --sku B1

# Create Web App for Containers
az webapp create \
  --resource-group $RG \
  --plan $PLAN_NAME \
  --name $WEBAPP_NAME \
  --deployment-container-image-name mcr.microsoft.com/appsvc/staticsite:latest

# Enable container logging
az webapp log config --resource-group $RG --name $WEBAPP_NAME --docker-container-logging filesystem

# Set port
az webapp config appsettings set --resource-group $RG --name $WEBAPP_NAME --settings WEBSITES_PORT=8000
```

## 2. Create Service Principal for GitHub Actions

Create a service principal with Contributor access scoped to the resource group:

```bash
az ad sp create-for-rbac \
  --name "github-actions-thesis" \
  --role contributor \
  --scopes /subscriptions/$(az account show --query id -o tsv)/resourceGroups/thesis-cicd-rg \
  --sdk-auth
```

This outputs JSON like:

```json
{
  "clientId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "clientSecret": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "subscriptionId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "tenantId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "activeDirectoryEndpointUrl": "https://login.microsoftonline.com",
  "resourceManagerEndpointUrl": "https://management.azure.com/",
  "activeDirectoryGraphResourceId": "https://graph.windows.net/",
  "sqlManagementEndpointUrl": "https://management.core.windows.net:8443/",
  "galleryEndpointUrl": "https://gallery.azure.com/",
  "managementEndpointUrl": "https://management.core.windows.net/"
}
```

**Copy this entire JSON output** - you'll need it for the next step.

## 3. Configure GitHub Secrets and Variables

In your GitHub repository, go to **Settings > Secrets and variables > Actions**.

### Repository Secret (required)

| Name | Value |
|------|-------|
| `AZURE_CREDENTIALS` | The entire JSON output from `az ad sp create-for-rbac --sdk-auth` |

### Repository Variables (required)

| Name | Example Value | Description |
|------|---------------|-------------|
| `ACR_LOGIN_SERVER` | `thesiscicdacr.azurecr.io` | ACR login server URL |
| `ACR_NAME` | `thesiscicdacr` | ACR name (without .azurecr.io) |
| `AZURE_WEBAPP_NAME` | `thesis-cicd-app` | Web App name |
| `AZURE_RG` | `thesis-cicd-rg` | Resource group name |

### How to get ACR login server

```bash
az acr show --name thesiscicdacr --query loginServer --output tsv
# Output: thesiscicdacr.azurecr.io
```

## 4. Verify Deployment

After pushing to `main`, the workflow will:
1. Run backend tests
2. Build frontend
3. Build Docker image
4. Push to ACR
5. Deploy to Azure Web App

Check your app at: `https://<WEBAPP_NAME>.azurewebsites.net`

### View logs

```bash
# Stream container logs
az webapp log tail --resource-group thesis-cicd-rg --name thesis-cicd-app

# View deployment logs in GitHub Actions artifacts
```

## Troubleshooting

### ACR login fails
Ensure the service principal has `AcrPush` role on the ACR:
```bash
az role assignment create \
  --assignee <clientId-from-sp> \
  --role AcrPush \
  --scope /subscriptions/<sub-id>/resourceGroups/<rg>/providers/Microsoft.ContainerRegistry/registries/<acr-name>
```

### Web App can't pull image
The setup script configures ACR admin credentials. If issues persist:
```bash
az webapp config container set \
  --resource-group $RG \
  --name $WEBAPP_NAME \
  --docker-registry-server-url https://$ACR_LOGIN_SERVER \
  --docker-registry-server-user $(az acr credential show --name $ACR_NAME --query username -o tsv) \
  --docker-registry-server-password $(az acr credential show --name $ACR_NAME --query "passwords[0].value" -o tsv)
```

### App returns 502/503
Check container logs and ensure `WEBSITES_PORT=8000` is set:
```bash
az webapp config appsettings list --resource-group $RG --name $WEBAPP_NAME
```

## Cleanup

To delete all resources (**destructive**):
```bash
az group delete --name thesis-cicd-rg --yes --no-wait
```
