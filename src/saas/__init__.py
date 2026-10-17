"""SaaS layer: authentication, tenants, private file storage, upload
validation, jobs and audit (see ``docs/security/saas-readiness.md``).

The browser and every uploaded file are untrusted: tenant and role come
from the server-side session, never from the request.
"""
