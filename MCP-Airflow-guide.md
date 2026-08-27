# Serveur MCP pour Apache Airflow 3.x — Guide d'architecture et d'implémentation

## ⚠️ Correction critique avant tout : impact de vos contraintes réseau

Votre contrainte réseau change fondamentalement l'architecture à retenir, et il faut la traiter **avant** de discuter transport, sécurité ou code. Voici pourquoi.

### Les deux mécanismes MCP de Claude Desktop ne sont pas équivalents

Claude Desktop expose **deux façons distinctes** de parler à un serveur MCP, et elles n'ont pas le même chemin réseau :

| Mécanisme | Où tourne le processus client | D'où part la connexion réseau | Transport |
|---|---|---|---|
| **Serveur local** (`claude_desktop_config.json`, clé `mcpServers`) | Sur le poste de travail, lancé par Claude Desktop | **Depuis le poste de travail lui-même** | STDIO (le serveur MCP doit parler STDIO ; s'il parle HTTP, il faut un pont local) |
| **Custom Connector / Remote MCP** (Settings → Connectors) | Rien sur le poste de travail | **Depuis l'infrastructure cloud d'Anthropic**, pas depuis votre PC | Streamable HTTP ou SSE, exposé publiquement |

C'est le point that beaucoup de gens ratent : même si Claude Desktop tourne sur votre PC, un connecteur distant (« remote MCP ») fait transiter les appels **par les serveurs d'Anthropic**, pas par la carte réseau de votre machine. Anthropic le documente explicitement : la connexion à un serveur MCP distant est brokée depuis leur infrastructure cloud, et les serveurs qui ne sont accessibles que depuis un réseau privé, derrière un VPN ou un pare-feu, **ne pourront pas être joints** par ce mécanisme.

### Conséquence directe sur vos 3 scénarios proposés

Vos scénarios 1, 2 et 3 supposent tous « Claude Desktop → MCP Server interne » en utilisant le transport Streamable HTTP comme un connecteur distant classique. Or :

- Votre VM MCP n'a **pas d'IP publique**.
- Les flux entrants depuis Internet vers vos VM sont **interdits**.
- Un reverse proxy interne (scénario 2) ou une API Gateway interne (scénario 3) ne change rien à ce problème tant qu'ils ne sont pas eux-mêmes exposés publiquement — ce que vous excluez.

**Résultat : aucun des scénarios 1, 2, 3 ne peut fonctionner via le mécanisme « Custom Connector / Remote MCP », quel que soit le nombre de proxys internes que vous ajoutez.** Le proxy ou la gateway devraient être joignables depuis les serveurs d'Anthropic sur Internet, ce qui viole votre politique.

### La solution : scénario 4, le pont local (bridge) sur le poste de travail

La bonne architecture exploite l'**autre** mécanisme : le serveur MCP **local** déclaré dans `claude_desktop_config.json`. Celui-ci est lancé comme un sous-processus **sur votre poste de travail**, qui fait lui-même partie du réseau interne autorisé à parler aux serveurs internes.

Le serveur MCP réel (Python, FastAPI/MCP SDK, transport Streamable HTTP) continue de tourner sur votre VM interne, **jamais exposé sur Internet**. Sur le poste de travail, Claude Desktop lance un petit processus pont STDIO ↔ HTTP (le paquet `mcp-remote`, ou un pont équivalent) qui :

1. Parle STDIO à Claude Desktop (conforme à ce que `claude_desktop_config.json` attend).
2. Traduit ces appels en requêtes Streamable HTTP vers `https://mcp-airflow.interne.entreprise.local:8443`.
3. Ces requêtes HTTP partent **du poste de travail**, donc respectent votre règle « poste de travail interne → serveurs internes ».

```
┌─────────────────────────────┐
│   Poste de travail (interne) │
│                              │
│  Claude Desktop              │
│    │ STDIO (subprocess)      │
│    ▼                         │
│  mcp-remote (bridge local)   │
└───────────────┬──────────────┘
                │ HTTPS / Streamable HTTP
                │ (poste → serveur interne : autorisé)
                ▼
┌─────────────────────────────────────┐
│  VM MCP Server (réseau interne)      │
│  Reverse proxy (nginx/Traefik) :443  │
│    │  mTLS ou Bearer token           │
│    ▼                                 │
│  Serveur MCP (FastMCP, uvicorn)      │
│    │  HTTPS interne + token API      │
│    ▼                                 │
└───────────────┬───────────────────────┘
                │ (serveur → serveur : autorisé)
                ▼
┌─────────────────────────────────────┐
│  Plateformes Airflow 3.x (REST API)  │
│  env-prod, env-preprod, env-dev...   │
└───────────────────────────────────────┘
```

Cette architecture ne viole aucune de vos règles : aucun flux entrant depuis Internet, aucune IP publique nécessaire, tout reste poste-de-travail-vers-interne ou serveur-vers-serveur.

### Réponses directes à vos questions de découverte/joignabilité

- **Comment Claude Desktop découvre le serveur MCP ?** Pas de découverte automatique. Vous déclarez explicitement l'entrée dans `claude_desktop_config.json` (nom, commande à lancer, arguments). Il n'y a pas de DNS-SD ni de mDNS impliqué.
- **Quels ports ouvrir ?** Uniquement le port du reverse proxy interne sur la VM MCP (443 typiquement), accessible **depuis les sous-réseaux des postes de travail uniquement** (pas depuis Internet, pas depuis tout le LAN si vous pouvez le restreindre par pare-feu/ACL). Aucun port entrant n'est nécessaire côté poste de travail.
- **Streamable HTTP est-il réellement supporté par Claude Desktop dans ce scénario ?** Oui, mais indirectement : Claude Desktop lui-même ne parle pas Streamable HTTP nativement pour les serveurs déclarés en local (il parle STDIO à ses sous-processus) ; c'est le pont `mcp-remote` qui parle Streamable HTTP à votre serveur. Le protocole Streamable HTTP est donc bien utilisé, mais entre le pont (sur le poste) et votre VM, pas entre Claude Desktop et votre VM directement.
- **Serveur distant vs serveur local : lequel privilégier ici ?** Le mécanisme « distant » (Custom Connector) est *impossible* compte tenu de vos contraintes réseau (il exigerait une exposition Internet). Le mécanisme « local avec pont » est donc la seule option viable, et elle est d'ailleurs la pratique recommandée par Anthropic pour les serveurs MCP internes/d'entreprise non exposés publiquement.
- **Limitations selon la version de Claude Desktop ?** Le pont `mcp-remote` nécessite Node.js installé sur le poste de travail. Les versions récentes de Claude Desktop valident strictement le schéma `mcpServers` (STDIO uniquement) — un champ `url`/`type: http` collé directement dans `claude_desktop_config.json` est ignoré ou fait échouer le chargement de la configuration. Vérifiez la version exacte de Claude Desktop utilisée par vos utilisateurs pour confirmer la présence du support `mcp-remote`/DXT au moment du déploiement, car ces mécanismes évoluent.

### Choix final entre vos 4 scénarios

| Scénario | Faisable avec vos contraintes ? |
|---|---|
| 1. Claude Desktop → MCP Server interne (direct, remote) | ❌ Non — nécessite exposition publique |
| 2. Claude Desktop → Reverse Proxy interne → MCP Server (remote) | ❌ Non — même problème, le proxy devrait être public |
| 3. Claude Desktop → API Gateway interne → MCP Server (remote) | ❌ Non — idem |
| **4. Claude Desktop → pont local (mcp-remote) → Reverse Proxy interne → MCP Server → Airflow** | ✅ **Oui — c'est l'architecture retenue pour la suite de ce document** |

Le reste de ce guide est bâti sur ce scénario 4.

---

## 1. Architecture détaillée

### 1.1 Composants

- **Claude Desktop** : client MCP graphique sur le poste de travail. Ne fait aucun appel réseau direct vers la VM — il délègue à un sous-processus.
- **mcp-remote (pont local)** : petit processus Node.js lancé par Claude Desktop via STDIO, qui ouvre une session Streamable HTTP vers l'URL interne du serveur MCP. Gère la conversion de protocole et, si besoin, le porteur du jeton d'authentification.
- **Reverse proxy interne (nginx ou Traefik)** sur la VM MCP : termine le TLS (certificat interne signé par votre PKI d'entreprise), applique l'authentification de première ligne (mTLS client ou vérification de jeton), limite le débit, journalise les accès.
- **Serveur MCP (Python, `mcp` SDK + FastAPI/uvicorn)** : implémente les *tools* MCP, contient la logique métier « traduire un tool MCP en appel API Airflow », gère le multi-environnement.
- **API REST Airflow 3.x** : une par plateforme Airflow (prod, préprod, dev, etc.), interrogée en HTTPS interne avec un compte de service dédié par environnement.
- **Authentification** : trois niveaux distincts à ne pas confondre — (a) poste de travail → proxy (mTLS ou token utilisateur), (b) proxy → serveur MCP (réseau de confiance, éventuellement token interne), (c) serveur MCP → Airflow (compte de service par environnement, jetons API Airflow).
- **Réseau** : aucun segment n'est traversé par un flux entrant depuis Internet ; tout est poste de travail → interne ou interne → interne.

### 1.2 Flux HTTP nominal (exemple `trigger_dag`)

1. Utilisateur, dans Claude App, demande de déclencher `dag_id=export_ventes` sur `env=prod`.
2. Claude Desktop invoque le tool MCP `trigger_dag` via STDIO sur le pont `mcp-remote`.
3. `mcp-remote` transforme cet appel en requête `POST` Streamable HTTP vers `https://mcp-airflow.interne.entreprise.local/mcp`, avec le jeton du poste de travail en en-tête `Authorization`.
4. Le reverse proxy vérifie le certificat client (mTLS) ou le jeton, puis relaie en interne (loopback ou réseau privé de la VM) vers uvicorn (`127.0.0.1:8000`).
5. Le serveur MCP résout le tool `trigger_dag`, retrouve la configuration de l'environnement `prod` (URL Airflow, credentials), obtient/rafraîchit un jeton Airflow si nécessaire.
6. Il appelle `POST /api/v2/dags/export_ventes/dagRuns` sur l'API Airflow prod.
7. La réponse Airflow est mise en forme en résultat de tool MCP et redescend par la même chaîne jusqu'à Claude Desktop.

---

## 2. Choix techniques et justification

| Choix | Pourquoi | Inconvénients à connaître |
|---|---|---|
| **Python + `mcp` SDK officiel** (`pip install mcp`) | SDK maintenu par Anthropic/communauté MCP, implémente nativement Streamable HTTP, gestion de session, découverte des tools/schemas JSON | Écosystème encore jeune, API a changé entre versions du spec MCP — figez la version |
| **`FastMCP`** (classe haut niveau du SDK `mcp`) | Décoration `@mcp.tool()` très simple, génère automatiquement le schéma JSON à partir des type hints Python | Moins de contrôle fin que l'API bas niveau (`Server`) si vous avez besoin de middlewares très spécifiques |
| **Streamable HTTP plutôt que SSE** | C'est le transport HTTP recommandé par la version courante du protocole MCP ; SSE est en cours de dépréciation | Nécessite un client compatible (c'est le rôle de `mcp-remote` ici) |
| **uvicorn comme serveur ASGI** | Standard, performant, bien supporté par Starlette/FastAPI sur lesquels le SDK MCP s'appuie | Un seul worker par défaut ; pour la charge, passer par plusieurs workers/processus (voir §7) |
| **Reverse proxy nginx/Traefik devant uvicorn** | Termine le TLS avec un certificat interne, ajoute mTLS si souhaité, journalise indépendamment de l'appli, permet le rate limiting | Complexité de configuration supplémentaire, mais standard dans toute architecture d'entreprise |
| **httpx (client async) pour appeler Airflow** | Cohérent avec le monde asyncio de FastAPI/MCP, gère HTTP/2, timeouts, pools de connexions | Nécessite une gestion explicite des erreurs réseau/timeout par environnement |
| **mcp-remote comme pont côté poste de travail** | Seule option qui rend Streamable HTTP joignable depuis un serveur interne non exposé publiquement, en respectant le modèle « serveur local » de Claude Desktop | Dépendance à Node.js sur le poste de travail ; alternative : empaqueter en extension Desktop (.dxt) pour simplifier la distribution aux utilisateurs |

---

## 3. Installation — commandes Linux complètes

Exécuté sur la VM Linux dédiée (exemple Debian/Ubuntu ; adaptez les noms de paquets pour RHEL/Rocky).

```bash
# 1. Mise à jour système
sudo apt update && sudo apt -y upgrade

# 2. Création d'un utilisateur système dédié (principe du moindre privilège)
sudo useradd --system --create-home --home-dir /opt/mcp-airflow \
    --shell /usr/sbin/nologin mcpairflow

# 3. Arborescence du projet
sudo mkdir -p /opt/mcp-airflow/{app,config,logs,certs,venv}
sudo chown -R mcpairflow:mcpairflow /opt/mcp-airflow
sudo chmod 750 /opt/mcp-airflow
sudo chmod 700 /opt/mcp-airflow/config /opt/mcp-airflow/certs   # secrets/credentials

# 4. Python et outils de build
sudo apt install -y python3.12 python3.12-venv python3-pip build-essential

# 5. Environnement virtuel (exécuté en tant qu'utilisateur mcpairflow)
sudo -u mcpairflow bash -c '
  cd /opt/mcp-airflow
  python3.12 -m venv venv
  source venv/bin/activate
  pip install --upgrade pip
'

# 6. Dépendances applicatives
sudo -u mcpairflow /opt/mcp-airflow/venv/bin/pip install \
    "mcp[cli]>=1.6" \
    "fastapi>=0.115" \
    "uvicorn[standard]>=0.30" \
    "httpx>=0.27" \
    "pydantic>=2.8" \
    "pydantic-settings>=2.4" \
    "python-json-logger>=2.0" \
    "tenacity>=9.0"

# 7. Figer les versions (reproductibilité)
sudo -u mcpairflow /opt/mcp-airflow/venv/bin/pip freeze | sudo -u mcpairflow tee /opt/mcp-airflow/requirements.lock.txt

# 8. Reverse proxy
sudo apt install -y nginx

# 9. Répertoire de configuration nginx dédié
sudo mkdir -p /etc/nginx/mcp-airflow
```

Structure finale attendue :

```
/opt/mcp-airflow/
├── venv/                     # environnement virtuel Python
├── app/                      # code source (voir §4)
│   ├── main.py
│   ├── settings.py
│   ├── airflow_client.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── consultation.py
│   │   ├── execution.py
│   │   └── administration.py
│   └── logging_config.py
├── config/
│   └── environments.yaml     # accès permission 600, propriétaire mcpairflow uniquement
├── certs/                    # certificats internes (clé privée : 600)
├── logs/
└── requirements.lock.txt
```

---

## 4. Développement du serveur MCP

### 4.1 Bonnes pratiques de structuration

- Séparer clairement **transport MCP** (déclaration des tools), **client Airflow** (appels REST) et **configuration** (multi-environnement).
- Aucun secret en dur dans le code : tout passe par `config/environments.yaml` (droits 600) ou par un coffre-fort de secrets (Vault, etc.) si disponible.
- Chaque tool MCP doit être **idempotent en lecture**, et pour les tools d'écriture (trigger, pause, kill…), journaliser systématiquement *qui* a demandé *quoi* sur *quel environnement*.
- Schémas de paramètres explicites (Pydantic) pour que Claude Desktop affiche des formulaires de tool clairs et validés côté serveur.

### 4.2 `app/settings.py`

```python
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml
from pathlib import Path


class AirflowEnvironment(BaseModel):
    name: str
    base_url: str            # ex: https://airflow-prod.interne.entreprise.local
    auth_username: str
    auth_password: str       # chargé depuis un fichier séparé en production
    verify_tls: bool = True
    ca_bundle: str | None = None   # chemin vers la CA interne si nécessaire


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MCP_AIRFLOW_")

    host: str = "127.0.0.1"     # uvicorn n'écoute qu'en local ; nginx fait la façade
    port: int = 8000
    log_level: str = "INFO"
    environments_file: str = "/opt/mcp-airflow/config/environments.yaml"

    def load_environments(self) -> dict[str, AirflowEnvironment]:
        data = yaml.safe_load(Path(self.environments_file).read_text())
        return {k: AirflowEnvironment(name=k, **v) for k, v in data["environments"].items()}


settings = Settings()
ENVIRONMENTS = settings.load_environments()
```

`config/environments.yaml` (droits `600`, jamais commité) :

```yaml
environments:
  prod:
    base_url: "https://airflow-prod.interne.entreprise.local"
    auth_username: "svc-mcp-airflow"
    auth_password: "REMPLACER_PAR_VALEUR_SECRETE"
    verify_tls: true
    ca_bundle: "/opt/mcp-airflow/certs/ca-interne.pem"
  preprod:
    base_url: "https://airflow-preprod.interne.entreprise.local"
    auth_username: "svc-mcp-airflow"
    auth_password: "REMPLACER_PAR_VALEUR_SECRETE"
    verify_tls: true
    ca_bundle: "/opt/mcp-airflow/certs/ca-interne.pem"
```

### 4.3 `app/airflow_client.py`

```python
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from .settings import AirflowEnvironment


class AirflowClientError(Exception):
    pass


class AirflowClient:
    """Client HTTP pour l'API REST officielle d'Airflow 3.x, un par environnement."""

    def __init__(self, env: AirflowEnvironment):
        self.env = env
        self._client = httpx.AsyncClient(
            base_url=env.base_url,
            verify=env.ca_bundle if env.ca_bundle else env.verify_tls,
            timeout=httpx.Timeout(10.0, read=30.0),
        )
        self._token: str | None = None

    async def _ensure_token(self) -> str:
        # Airflow 3.x : authentification par jeton via le endpoint d'auth du auth manager configuré.
        # Adaptez le chemin exact selon votre Auth Manager (Fab, Simple, etc.) — à vérifier
        # dans la configuration de chaque plateforme Airflow avant mise en production.
        if self._token:
            return self._token
        resp = await self._client.post(
            "/auth/token",
            json={"username": self.env.auth_username, "password": self.env.auth_password},
        )
        resp.raise_for_status()
        self._token = resp.json()["access_token"]
        return self._token

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    async def request(self, method: str, path: str, **kwargs) -> dict:
        token = await self._ensure_token()
        headers = kwargs.pop("headers", {})
        headers["Authorization"] = f"Bearer {token}"
        resp = await self._client.request(method, path, headers=headers, **kwargs)
        if resp.status_code == 401:
            # jeton expiré : on force un renouvellement puis on retente une fois
            self._token = None
            token = await self._ensure_token()
            headers["Authorization"] = f"Bearer {token}"
            resp = await self._client.request(method, path, headers=headers, **kwargs)
        if resp.status_code >= 400:
            raise AirflowClientError(f"{method} {path} -> {resp.status_code}: {resp.text}")
        return resp.json() if resp.content else {}

    async def aclose(self):
        await self._client.aclose()
```

### 4.4 `app/main.py` — démarrage du serveur MCP en Streamable HTTP

```python
import logging
from mcp.server.fastmcp import FastMCP

from .settings import settings, ENVIRONMENTS
from .airflow_client import AirflowClient
from .logging_config import configure_logging
from .tools import consultation, execution, administration

configure_logging(settings.log_level)
logger = logging.getLogger("mcp-airflow")

mcp = FastMCP("airflow-mcp-server")

# Un client HTTP par environnement Airflow, réutilisé entre appels
CLIENTS: dict[str, AirflowClient] = {
    name: AirflowClient(env) for name, env in ENVIRONMENTS.items()
}

consultation.register_tools(mcp, CLIENTS)
execution.register_tools(mcp, CLIENTS)
administration.register_tools(mcp, CLIENTS)


if __name__ == "__main__":
    logger.info("Démarrage du serveur MCP Airflow sur %s:%s", settings.host, settings.port)
    # Le SDK MCP expose directement le transport Streamable HTTP.
    # uvicorn n'écoute qu'en local (127.0.0.1) : le reverse proxy fait la façade TLS/mTLS.
    mcp.run(transport="streamable-http", host=settings.host, port=settings.port)
```

### 4.5 `app/logging_config.py`

```python
import logging
from pythonjsonlogger import jsonlogger


def configure_logging(level: str = "INFO"):
    handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
```

### 4.6 Gestion des erreurs — convention commune

Chaque tool doit intercepter `AirflowClientError` et retourner un message structuré et exploitable par Claude, plutôt que de laisser fuiter une trace Python brute :

```python
from ..airflow_client import AirflowClientError

async def safe_call(coro):
    try:
        return await coro
    except AirflowClientError as e:
        logger.warning("Appel Airflow échoué: %s", e)
        return {"error": True, "message": str(e)}
```

---

## 5. Intégration Airflow — un tool complet par fonctionnalité

### 5.1 `app/tools/consultation.py`

```python
import logging
from mcp.server.fastmcp import FastMCP
from ..airflow_client import AirflowClient, AirflowClientError

logger = logging.getLogger("mcp-airflow.consultation")


def register_tools(mcp: FastMCP, clients: dict[str, AirflowClient]):

    @mcp.tool()
    async def list_environments() -> list[str]:
        """Liste les environnements Airflow disponibles pour ce serveur MCP."""
        return list(clients.keys())

    @mcp.tool()
    async def list_dags(env: str, only_active: bool = True) -> dict:
        """Liste les DAGs d'un environnement Airflow donné."""
        client = clients[env]
        params = {"only_active": only_active}
        try:
            return await client.request("GET", "/api/v2/dags", params=params)
        except AirflowClientError as e:
            return {"error": True, "message": str(e)}

    @mcp.tool()
    async def get_dag(env: str, dag_id: str) -> dict:
        """Détail d'un DAG spécifique."""
        client = clients[env]
        try:
            return await client.request("GET", f"/api/v2/dags/{dag_id}")
        except AirflowClientError as e:
            return {"error": True, "message": str(e)}

    @mcp.tool()
    async def get_dag_runs(env: str, dag_id: str, limit: int = 10) -> dict:
        """Derniers DAG Runs pour un DAG donné."""
        client = clients[env]
        params = {"limit": limit, "order_by": "-start_date"}
        try:
            return await client.request("GET", f"/api/v2/dags/{dag_id}/dagRuns", params=params)
        except AirflowClientError as e:
            return {"error": True, "message": str(e)}

    @mcp.tool()
    async def get_dag_run_tasks(env: str, dag_id: str, dag_run_id: str) -> dict:
        """Liste des instances de tâches pour un DAG Run donné."""
        client = clients[env]
        try:
            return await client.request(
                "GET", f"/api/v2/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances"
            )
        except AirflowClientError as e:
            return {"error": True, "message": str(e)}

    @mcp.tool()
    async def get_task_logs(
        env: str, dag_id: str, dag_run_id: str, task_id: str, try_number: int = 1
    ) -> dict:
        """Logs d'une tâche pour une tentative donnée."""
        client = clients[env]
        try:
            result = await client.request(
                "GET",
                f"/api/v2/dags/{dag_id}/dagRuns/{dag_run_id}"
                f"/taskInstances/{task_id}/logs/{try_number}",
            )
            return result
        except AirflowClientError as e:
            return {"error": True, "message": str(e)}

    @mcp.tool()
    async def list_workers(env: str) -> dict:
        """Informations sur les workers (via l'endpoint santé/monitoring d'Airflow)."""
        client = clients[env]
        try:
            return await client.request("GET", "/api/v2/monitor/health")
        except AirflowClientError as e:
            return {"error": True, "message": str(e)}

    @mcp.tool()
    async def list_pools(env: str) -> dict:
        """Liste des pools Airflow et leur utilisation."""
        client = clients[env]
        try:
            return await client.request("GET", "/api/v2/pools")
        except AirflowClientError as e:
            return {"error": True, "message": str(e)}

    @mcp.tool()
    async def list_variables(env: str) -> dict:
        """Liste des variables Airflow (les valeurs sensibles doivent être masquées côté Airflow)."""
        client = clients[env]
        try:
            return await client.request("GET", "/api/v2/variables")
        except AirflowClientError as e:
            return {"error": True, "message": str(e)}

    @mcp.tool()
    async def list_connections(env: str) -> dict:
        """Liste des connexions Airflow déclarées (mots de passe masqués par l'API Airflow elle-même)."""
        client = clients[env]
        try:
            return await client.request("GET", "/api/v2/connections")
        except AirflowClientError as e:
            return {"error": True, "message": str(e)}
```

> Remarque sécurité : `list_variables` et `list_connections` peuvent exposer des informations sensibles même si Airflow masque les secrets par défaut. Envisagez de restreindre ces deux tools par un rôle dédié (voir §6, RBAC applicatif).

### 5.2 `app/tools/execution.py`

```python
import logging
from mcp.server.fastmcp import FastMCP
from ..airflow_client import AirflowClient, AirflowClientError

logger = logging.getLogger("mcp-airflow.execution")


def register_tools(mcp: FastMCP, clients: dict[str, AirflowClient]):

    @mcp.tool()
    async def trigger_dag(env: str, dag_id: str, conf: dict | None = None) -> dict:
        """Déclenche un nouveau DAG Run."""
        client = clients[env]
        logger.info("trigger_dag env=%s dag_id=%s", env, dag_id)
        try:
            return await client.request(
                "POST", f"/api/v2/dags/{dag_id}/dagRuns",
                json={"conf": conf or {}},
            )
        except AirflowClientError as e:
            return {"error": True, "message": str(e)}

    @mcp.tool()
    async def clear_dag_run(env: str, dag_id: str, dag_run_id: str) -> dict:
        """Relance un DAG Run en nettoyant l'état de ses tâches."""
        client = clients[env]
        logger.info("clear_dag_run env=%s dag_id=%s run=%s", env, dag_id, dag_run_id)
        try:
            return await client.request(
                "POST", f"/api/v2/dags/{dag_id}/clearTaskInstances",
                json={"dag_run_id": dag_run_id, "dry_run": False},
            )
        except AirflowClientError as e:
            return {"error": True, "message": str(e)}

    @mcp.tool()
    async def clear_task_instance(
        env: str, dag_id: str, dag_run_id: str, task_id: str
    ) -> dict:
        """Relance une tâche spécifique au sein d'un DAG Run."""
        client = clients[env]
        logger.info(
            "clear_task_instance env=%s dag_id=%s run=%s task=%s",
            env, dag_id, dag_run_id, task_id,
        )
        try:
            return await client.request(
                "POST", f"/api/v2/dags/{dag_id}/clearTaskInstances",
                json={
                    "dag_run_id": dag_run_id,
                    "task_ids": [task_id],
                    "dry_run": False,
                },
            )
        except AirflowClientError as e:
            return {"error": True, "message": str(e)}

    @mcp.tool()
    async def stop_dag_run(env: str, dag_id: str, dag_run_id: str) -> dict:
        """Arrête (marque comme failed) un DAG Run en cours."""
        client = clients[env]
        logger.info("stop_dag_run env=%s dag_id=%s run=%s", env, dag_id, dag_run_id)
        try:
            return await client.request(
                "PATCH", f"/api/v2/dags/{dag_id}/dagRuns/{dag_run_id}",
                json={"state": "failed"},
            )
        except AirflowClientError as e:
            return {"error": True, "message": str(e)}

    @mcp.tool()
    async def pause_dag(env: str, dag_id: str) -> dict:
        """Met un DAG en pause."""
        client = clients[env]
        logger.info("pause_dag env=%s dag_id=%s", env, dag_id)
        try:
            return await client.request(
                "PATCH", f"/api/v2/dags/{dag_id}", json={"is_paused": True}
            )
        except AirflowClientError as e:
            return {"error": True, "message": str(e)}

    @mcp.tool()
    async def unpause_dag(env: str, dag_id: str) -> dict:
        """Réactive un DAG en pause."""
        client = clients[env]
        logger.info("unpause_dag env=%s dag_id=%s", env, dag_id)
        try:
            return await client.request(
                "PATCH", f"/api/v2/dags/{dag_id}", json={"is_paused": False}
            )
        except AirflowClientError as e:
            return {"error": True, "message": str(e)}
```

### 5.3 `app/tools/administration.py`

```python
import asyncio
import logging
from mcp.server.fastmcp import FastMCP
from ..airflow_client import AirflowClient, AirflowClientError

logger = logging.getLogger("mcp-airflow.administration")


def register_tools(mcp: FastMCP, clients: dict[str, AirflowClient]):

    @mcp.tool()
    async def health_check(env: str) -> dict:
        """Santé détaillée de la plateforme : webserver, scheduler, triggerer, base."""
        client = clients[env]
        try:
            return await client.request("GET", "/api/v2/monitor/health")
        except AirflowClientError as e:
            return {"error": True, "message": str(e)}

    @mcp.tool()
    async def global_status_summary(env: str) -> dict:
        """Synthèse globale : santé + DAGs actifs + derniers échecs, en un seul appel."""
        client = clients[env]
        try:
            health, dags = await asyncio.gather(
                client.request("GET", "/api/v2/monitor/health"),
                client.request("GET", "/api/v2/dags", params={"only_active": True}),
            )
            return {
                "environment": env,
                "health": health,
                "active_dags_count": dags.get("total_entries", 0),
            }
        except AirflowClientError as e:
            return {"error": True, "message": str(e)}

    @mcp.tool()
    async def global_status_all_environments() -> dict:
        """Synthèse de santé pour tous les environnements enregistrés."""
        results = {}
        for name, client in clients.items():
            try:
                results[name] = await client.request("GET", "/api/v2/monitor/health")
            except AirflowClientError as e:
                results[name] = {"error": True, "message": str(e)}
        return results
```

> Les chemins exacts de l'API REST Airflow 3.x (`/api/v2/...`) et le format précis de l'endpoint de santé/authentification dépendent de la version mineure et du plugin d'Auth Manager configuré (Fab Auth Manager, Simple Auth Manager, etc.). Vérifiez-les contre la documentation OpenAPI exposée par chacune de vos plateformes (`/openapi.json` ou l'interface Swagger habituellement disponible) avant la mise en production, et ajustez le client en conséquence.

---

## 6. Sécurité — détail par couche

- **TLS** : certificat interne signé par votre PKI d'entreprise sur le reverse proxy (`nginx`), TLS 1.2+ minimum, chiffrements modernes uniquement. Rotation automatisée du certificat (script + cron, ou intégration à votre outil de PKI interne type Vault/step-ca).
- **Certificats internes** : distribuez la CA interne sur le poste de travail (magasin de certificats du système) pour que `mcp-remote` valide la chaîne sans désactiver la vérification TLS. Ne désactivez jamais `verify_tls` en production, y compris en interne.
- **Authentification client (poste de travail → proxy)** : privilégiez le mTLS (certificat client par utilisateur ou par poste, émis par votre PKI) plutôt qu'un jeton statique, ce qui donne une traçabilité nominative forte. À défaut, un jeton porteur (Bearer) individuel avec expiration courte et renouvellement.
- **RBAC** : deux niveaux à distinguer.
  - RBAC applicatif dans le serveur MCP lui-même : certains tools (`trigger_dag`, `pause_dag`, `stop_dag_run`, `list_variables`, `list_connections`) peuvent être réservés à un sous-ensemble d'identités (vérification du sujet du certificat client ou du claim du jeton avant d'exécuter le tool).
  - RBAC Airflow : le compte de service `svc-mcp-airflow` par environnement doit avoir le rôle minimal nécessaire (lecture seule pour un environnement de reporting, opérateur pour un environnement où le déclenchement est autorisé).
- **Limitation des droits (moindre privilège)** :
  - Utilisateur système `mcpairflow` sans shell interactif (`nologin`).
  - Fichier `environments.yaml` en `600`, propriétaire `mcpairflow` uniquement.
  - uvicorn n'écoute que sur `127.0.0.1`, jamais directement exposé, même en interne.
  - Comptes de service Airflow distincts par environnement, jamais un compte administrateur global partagé.
- **Audit** : journaliser systématiquement, pour chaque tool d'écriture, l'identité appelante (sujet du certificat client ou claim du jeton), le tool, les paramètres, l'horodatage et le résultat. Conserver ces logs séparément des logs applicatifs génériques, avec une rétention conforme à votre politique interne.
- **Journalisation** : format JSON structuré (voir `logging_config.py`), niveau `INFO` en production, `DEBUG` réservé au dépannage temporaire (attention aux données sensibles dans les payloads Airflow).

---

## 7. Déploiement — Systemd, Docker, Kubernetes

### Recommandation pour une VM Linux dédiée : **Docker (via docker compose) piloté par systemd**

Justification : une VM unique et dédiée ne justifie pas la complexité opérationnelle de Kubernetes (etcd, control plane, réseau overlay) pour un seul service. Systemd seul fonctionne mais rend la gestion des dépendances/versions moins reproductible qu'un conteneur. Docker + systemd combine le meilleur des deux : image reproductible et versionnée, démarrage/arrêt/relance gérés nativement par systemd (intégration aux logs système, redémarrage automatique).

`Dockerfile` :

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.lock.txt .
RUN pip install --no-cache-dir -r requirements.lock.txt
COPY app/ ./app/
USER 1000:1000
EXPOSE 8000
CMD ["python", "-m", "app.main"]
```

`docker-compose.yml` :

```yaml
services:
  mcp-airflow:
    build: .
    restart: unless-stopped
    volumes:
      - ./config:/app/config:ro
      - ./certs:/app/certs:ro
    ports:
      - "127.0.0.1:8000:8000"
    environment:
      - MCP_AIRFLOW_LOG_LEVEL=INFO
```

Unité systemd `/etc/systemd/system/mcp-airflow.service` :

```ini
[Unit]
Description=Serveur MCP Airflow (via Docker Compose)
After=network-online.target docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/mcp-airflow
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
User=mcpairflow

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now mcp-airflow.service
```

Si votre organisation possède déjà un cluster Kubernetes géré et que ce service doit un jour scaler ou être colocalisé avec d'autres services internes, migrer vers un `Deployment` + `Service` ClusterIP (jamais un `LoadBalancer`/`Ingress` public) est parfaitement raisonnable — mais ce n'est pas nécessaire pour démarrer sur une VM dédiée.

---

## 8. Déclaration côté Claude Desktop

Compte tenu de la correction du §0, la déclaration se fait via le mécanisme **serveur local**, en utilisant `mcp-remote` comme pont vers votre URL Streamable HTTP interne.

Prérequis sur le poste de travail : Node.js et npm installés.

`claude_desktop_config.json` (emplacement standard : `%APPDATA%\Claude\claude_desktop_config.json` sous Windows, `~/Library/Application Support/Claude/claude_desktop_config.json` sous macOS) :

```json
{
  "mcpServers": {
    "airflow-interne": {
      "command": "npx",
      "args": [
        "-y",
        "mcp-remote",
        "https://mcp-airflow.interne.entreprise.local/mcp",
        "--header",
        "Authorization: Bearer ${MCP_AIRFLOW_TOKEN}"
      ]
    }
  }
}
```

Points d'attention :

- Utilisez le chemin complet de `npx` si Claude Desktop échoue à le trouver (PATH minimal au lancement) : remplacez `"command": "npx"` par le chemin absolu (`/usr/local/bin/npx` ou équivalent Windows).
- Le jeton peut être injecté via une variable d'environnement plutôt qu'écrit en clair dans le fichier, selon les capacités de substitution disponibles dans votre version de Claude Desktop — vérifiez ce point avant le déploiement à grande échelle et, à défaut, distribuez des jetons par utilisateur via un canal sécurisé plutôt que de les coder en dur.
- Après modification du fichier, redémarrez complètement Claude Desktop.
- Vérifiez dans l'onglet **Developer** de Claude Desktop que le serveur `airflow-interne` apparaît connecté et que la liste des tools se charge.

---

## 9. Validation — procédure de tests

1. **Handshake MCP** : depuis le poste de travail, testez le pont isolément avant de passer par Claude Desktop :
   ```bash
   npx -y mcp-remote https://mcp-airflow.interne.entreprise.local/mcp --header "Authorization: Bearer $TOKEN"
   ```
   Vérifiez que la session s'initialise sans erreur TLS ni 401.
2. **Découverte des tools** : dans Claude Desktop, ouvrez le panneau des tools disponibles pour le serveur `airflow-interne` et confirmez que les 17 tools (consultation + exécution + administration) apparaissent avec leurs schémas.
3. **Appels de tools** : testez dans l'ordre `list_environments`, `list_dags(env="dev")`, `get_dag(...)`, `health_check(env="dev")` sur un environnement non critique avant de valider sur `preprod`/`prod`.
4. **Tests d'erreurs** : environnement inconnu, `dag_id` inexistant, jeton expiré (attendez son expiration ou révoquez-le manuellement côté Airflow), coupure réseau simulée (bloquez temporairement le port sortant) — vérifiez que chaque cas retourne un message structuré et non une exception brute.
5. **Tests de performance** : mesurez la latence bout-en-bout (Claude → réponse) sur `list_dags` et `get_task_logs` (potentiellement volumineux) sous charge modérée (quelques appels concurrents), et fixez des timeouts cohérents côté `httpx` et côté reverse proxy.

---

## 10. Exploitation

- **Supervision** : exposez `/health` côté serveur MCP (endpoint FastAPI simple, séparé du protocole MCP) pour un monitoring externe (Prometheus blackbox exporter, Nagios/Centreon, etc.).
- **Métriques** : nombre d'appels par tool, taux d'erreur par environnement Airflow, latence p50/p95 — via un middleware ASGI exportant au format Prometheus.
- **Logs** : centralisez les journaux JSON du conteneur (pilote `journald` de Docker, puis Filebeat/Fluent Bit vers votre stack de logs interne).
- **Rotation des logs** : si vous ne centralisez pas immédiatement, configurez `logrotate` sur `/opt/mcp-airflow/logs` ou limitez la taille des logs Docker (`max-size`/`max-file` dans le driver `json-file`).
- **Sauvegarde** : le service est sans état persistant propre (l'état vit dans Airflow) ; sauvegardez uniquement `config/environments.yaml`, les certificats et la configuration nginx.
- **Montée de version** : figez les versions dans `requirements.lock.txt`, testez chaque mise à jour du SDK `mcp` (le protocole évolue vite) sur un environnement de dev avant `prod`, et gardez un tag Docker précédent disponible pour rollback immédiat.

---

## 11. Haute disponibilité

Avec un seul consommateur (votre client Claude Desktop, usage individuel ou petite équipe), une HA active/active est rarement justifiée dans un premier temps : la charge est faible et sporadique. Si la disponibilité devient critique (plusieurs équipes dépendantes) :

- Déployez deux instances du serveur MCP sur deux VM, derrière un load balancer L7 interne (le même reverse proxy peut jouer ce rôle, ou un LB dédié type HAProxy/F5 interne).
- Le serveur MCP tel que conçu ici est **sans état applicatif** (chaque appel Airflow est indépendant), donc l'équilibrage de charge simple (round-robin) fonctionne sans affinité de session particulière pour les *tool calls* eux-mêmes.
- Point d'attention : le protocole MCP Streamable HTTP peut maintenir un identifiant de session pour certaines fonctionnalités avancées (reprise de flux). Si vous en dépendez, ajoutez l'affinité de session (sticky sessions) sur le LB, ou partagez l'état de session via un magasin externe (Redis) entre les instances.
- **Impact côté Claude Desktop** : aucun changement de configuration — le pont `mcp-remote` continue de pointer vers une seule URL logique (celle du LB interne), qui distribue ensuite vers les instances saines.

---

## 12. Architecture recommandée pour la production

**Schéma cible** : celui du §0/§1 (scénario 4), avec en plus pour la production :

- Deux VM MCP derrière un LB interne (si criticité justifiée, sinon une seule VM avec systemd/Docker suffit initialement).
- Certificats internes émis et automatiquement renouvelés par votre PKI d'entreprise (éviter les certificats manuels à échéance oubliée).
- mTLS entre poste de travail et reverse proxy pour une traçabilité nominative forte.
- RBAC applicatif activé pour distinguer utilisateurs « lecture seule » et utilisateurs « opérateurs » (habilités à `trigger_dag`, `pause_dag`, etc.).
- Comptes de service Airflow dédiés et à droits minimaux, un par environnement, jamais partagés entre le MCP et d'autres usages.
- Centralisation des logs et des métriques dès la mise en production, pas en réaction à un premier incident.
- Procédure de rollback testée (image Docker taggée + configuration versionnée) avant chaque montée de version du SDK MCP.

**Dimensionnement indicatif pour un usage interne modéré** (quelques dizaines d'utilisateurs, usage interactif et non massivement concurrent) : 2 vCPU / 2 Go RAM par instance suffisent largement, le travail réel étant délégué à Airflow ; le serveur MCP n'est qu'un traducteur de protocole à faible empreinte.

**Recommandation d'exploitation finale** : commencez en configuration simple — une VM, Docker + systemd, un seul environnement Airflow connecté — validez la chaîne complète (pont local → proxy → serveur MCP → Airflow) de bout en bout avec un utilisateur pilote, puis étendez progressivement aux autres environnements Airflow et, si le besoin de disponibilité l'exige, à une seconde instance derrière un LB interne.
