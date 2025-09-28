"""
Integration tests for BMS Agent.

Test Coverage:
- Health checks
- Authentication
- Document search
- Document upload
- Error handling
- Edge cases
"""
import sys
import os
import pytest
import json
import asyncio
import httpx
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Test configuration
API_URL = "http://localhost:8000"
TEST_API_KEY = "test-key-123"  # Should match the one in your test environment

@pytest.mark.asyncio
async def test_health_check():
    """Test basic health check endpoint."""
    async with httpx.AsyncClient() as client:
        # Test basic health check
        response = await client.get(f"{API_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        
        # Test with invalid endpoint
        response = await client.get(f"{API_URL}/nonexistent")
        assert response.status_code == 404

@pytest.mark.asyncio
async def test_detailed_health():
    """Test detailed health check with service status."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{API_URL}/health/detailed",
            headers={"X-API-Key": TEST_API_KEY}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["api"] == "healthy"
        assert isinstance(data["qdrant"], bool)
        assert isinstance(data["collection_exists"], bool)

@pytest.mark.asyncio
async def test_search_endpoint():
    """Test the semantic search endpoint with various scenarios."""
    test_cases = [
        # (query, limit, expected_status, test_description)
        ("railway configuration", 3, 200, "valid query"),
        ("", 3, 400, "empty query"),
        ("a" * 1001, 3, 400, "query too long"),
        ("valid", 0, 400, "zero limit"),
        ("valid", 101, 400, "limit too high"),
        ("valid", -1, 400, "negative limit"),
    ]
    
    async with httpx.AsyncClient() as client:
        # Test without API key (should fail)
        response = await client.post(
            f"{API_URL}/api/v1/search/semantic",
            json={"query": "test", "limit": 3}
        )
        assert response.status_code == 403
        
        # Test with invalid API key
        response = await client.post(
            f"{API_URL}/api/v1/search/semantic",
            json={"query": "test", "limit": 3},
            headers={"X-API-Key": "invalid-key"}
        )
        assert response.status_code == 403
        
        # Test various query scenarios
        for query, limit, expected_status, desc in test_cases:
            response = await client.post(
                f"{API_URL}/api/v1/search/semantic",
                json={"query": query, "limit": limit},
                headers={"X-API-Key": TEST_API_KEY}
            )
            assert response.status_code == expected_status, f"Failed test case: {desc}"
            
            if expected_status == 200:
                data = response.json()
                assert "results" in data
                assert "count" in data
                assert isinstance(data["results"], list)
                assert isinstance(data["count"], int)
                assert len(data["results"]) <= limit

@pytest.mark.asyncio
async def test_upload_endpoint(tmp_path):
    """Test document upload and processing with various file types and edge cases."""
    # Create test files
    test_files = [
        ("test_document.txt", "text/plain", "This is a test document."),
        ("test.csv", "text/csv", "id,name\n1,test\n2,document"),
        ("test.pdf", "application/pdf", "%PDF-test"),  # Minimal PDF header
    ]
    
    for filename, content_type, content in test_files:
        test_file = tmp_path / filename
        test_file.write_text(content)
        
        async with httpx.AsyncClient() as client:
            # Test valid upload
            with open(test_file, "rb") as f:
                files = {"file": (filename, f, content_type)}
                response = await client.post(
                    f"{API_URL}/api/v1/documents/upload",
                    files=files,
                    headers={"X-API-Key": TEST_API_KEY}
                )
                
                assert response.status_code == 200, f"Failed to upload {filename}"
                data = response.json()
                assert data["status"] == "success"
                assert data["filename"] == filename
                assert "document_id" in data
                assert "chunks_processed" in data
                assert isinstance(data["chunks_processed"], int)
    
    # Test invalid file types
    invalid_files = [
        ("test.exe", "application/octet-stream", b"MZ", "Executable file"),
        ("test.bat", "application/bat", b"@echo off", "Batch file"),
    ]
    
    for filename, content_type, content, desc in invalid_files:
        test_file = tmp_path / filename
        test_file.write_bytes(content)
        
        async with httpx.AsyncClient() as client:
            with open(test_file, "rb") as f:
                files = {"file": (filename, f, content_type)}
                response = await client.post(
                    f"{API_URL}/api/v1/documents/upload",
                    files=files,
                    headers={"X-API-Key": TEST_API_KEY}
                )
                
                assert response.status_code == 400, f"Should reject {desc}"

    # Test file size limit (101MB)
    large_file = tmp_path / "large_file.bin"
    large_file.write_bytes(b"0" * (101 * 1024 * 1024))  # 101MB
    
    async with httpx.AsyncClient() as client:
        with open(large_file, "rb") as f:
            files = {"file": ("large_file.bin", f, "application/octet-stream")}
            response = await client.post(
                f"{API_URL}/api/v1/documents/upload",
                files=files,
                headers={"X-API-Key": TEST_API_KEY}
            )
            
            assert response.status_code == 413, "Should reject file over 100MB"

# Test error handling and edge cases

async def test_error_handling():
    """Test error handling for various edge cases."""
    async with httpx.AsyncClient() as client:
        # Test malformed JSON
        response = await client.post(
            f"{API_URL}/api/v1/search/semantic",
            content="{\"query\": \"test",  # Malformed JSON
            headers={"Content-Type": "application/json", "X-API-Key": TEST_API_KEY}
        )
        assert response.status_code == 422  # Unprocessable Entity
        
        # Test invalid content type
        response = await client.post(
            f"{API_URL}/api/v1/search/semantic",
            content="not json",
            headers={"Content-Type": "text/plain", "X-API-Key": TEST_API_KEY}
        )
        assert response.status_code == 415  # Unsupported Media Type

# Mock tests for service failures
async def test_service_failures():
    """Test behavior when dependent services fail."""
    with patch('api.main.processor.qdrant.get_collections') as mock_collections:
        # Simulate Qdrant connection failure
        mock_collections.side_effect = Exception("Connection failed")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{API_URL}/health/detailed",
                headers={"X-API-Key": TEST_API_KEY}
            )
            assert response.status_code == 503  # Service Unavailable

# Test coverage metrics and reporting
if __name__ == "__main__":
    """Run tests and generate coverage report."""
    import asyncio
    import pytest
    import sys
    
    async def run_tests():
        # Run all tests with coverage
        test_args = [
            "-v",
            "--cov=./",
            "--cov-report=term-missing",
            "--cov-report=html:coverage_html",
            "tests/"
        ]
        
        # Run pytest programmatically
        exit_code = pytest.main(test_args)
        
        # Generate coverage report
        print("\nTest coverage report:")
        print("-------------------")
        print("Coverage HTML report: file:///coverage_html/index.html")
        
        return exit_code
    
    # Run the tests
    exit_code = asyncio.run(run_tests())
    sys.exit(exit_code)
