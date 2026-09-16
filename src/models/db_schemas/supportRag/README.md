## Run Alembic Migrations

### Configuration

```bash
cp alembic.ini.example alembic.ini
```

- Update the `alembic.ini` with your database credentials (`sqlalchemy.url`)
  
### (Optional) Create a new migration

```bash
alembic revision --autogenerate -m "Add ..."
```

### Apply latest migrations

```bash
alembic upgrade head
```

### Undo the last migration (go back one version)

```bash
alembic downgrade -1
```

### See migration history

```bash
alembic history
```