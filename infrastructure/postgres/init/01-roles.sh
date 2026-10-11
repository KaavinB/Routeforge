#!/bin/sh
# Least-privilege database roles (runs once, on first container start).
#   POSTGRES_USER (routeforge_owner): owns the schema; used only by the migration job.
#   routeforge_app: the gateway's runtime role; SELECT/INSERT/UPDATE/DELETE only, no DDL.
set -eu
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<SQL
CREATE ROLE routeforge_app LOGIN PASSWORD '${ROUTEFORGE_APP_DB_PASSWORD}';
REVOKE ALL ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO routeforge_app;
ALTER DEFAULT PRIVILEGES FOR ROLE ${POSTGRES_USER} IN SCHEMA public
  GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO routeforge_app;
GRANT CONNECT ON DATABASE ${POSTGRES_DB} TO routeforge_app;
SQL
