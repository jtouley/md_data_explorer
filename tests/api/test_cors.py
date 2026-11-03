"""Tests for CORS configuration."""

import pytest


class TestCORSConfiguration:
    """Tests for CORS middleware configuration."""

    def test_cors_allows_electron_dev_origin(self, test_client):
        """CORS headers allow Electron dev server origin (localhost:5173)."""
        # Act: Make request with Electron dev server Origin header
        response = test_client.get(
            "/health",
            headers={"Origin": "http://localhost:5173"},
        )

        # Assert: Response includes CORS headers for Electron origin
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"

    def test_cors_packagedAppOrigin_preflightWithAuthorizationAllowed(self, test_client):
        """The packaged app (app:// scheme) can send the Authorization header cross-origin."""
        # Act
        response = test_client.options(
            "/api/datasets",
            headers={
                "Origin": "app://clinical-analytics",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "authorization",
            },
        )

        # Assert
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "app://clinical-analytics"
        assert "access-control-allow-credentials" not in response.headers

    @pytest.mark.parametrize("origin", ["null", "file://", "https://evil.example", "http://localhost:8000"])
    def test_cors_untrustedOrigin_notEchoed(self, test_client, origin):
        """file:// pages (Origin null), other sites and the API's own origin get no CORS grant."""
        # Act
        response = test_client.get("/health", headers={"Origin": origin})

        # Assert
        assert response.headers.get("access-control-allow-origin") is None

    @pytest.mark.parametrize("raw", ["null", "app://clinical-analytics,null", "*"])
    def test_corsOrigins_nullOrWildcardConfigured_raisesValueError(self, monkeypatch, raw):
        """A config that would let any file:// page or site call the API is refused at startup."""
        # Arrange
        from clinical_analytics.api.main import cors_origins

        monkeypatch.setenv("CORS_ORIGINS", raw)

        # Act / Assert
        with pytest.raises(ValueError, match="null"):
            cors_origins()

    def test_cors_preflight_allows_electron_origin(self, test_client):
        """CORS preflight (OPTIONS) allows Electron origin with required headers."""
        # Act: Make OPTIONS preflight request
        response = test_client.options(
            "/api/datasets",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )

        # Assert: Preflight response allows the origin
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
        assert "GET" in response.headers.get("access-control-allow-methods", "")

    def test_cors_still_allows_nextjs_origin(self, test_client):
        """CORS still allows existing Next.js origin (backward compatibility)."""
        # Act: Make request with Next.js Origin header
        response = test_client.get(
            "/health",
            headers={"Origin": "http://localhost:3000"},
        )

        # Assert: Response includes CORS headers for Next.js origin
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
