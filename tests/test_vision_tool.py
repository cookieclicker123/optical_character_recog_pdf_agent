import pytest
from pathlib import Path
from src.tools.vision_tool import setup_vision_tool
from src.data_model import VisionTool
from src.prompts.post_prompts import EXTRACT_PROMPT

@pytest.fixture
def vision_tool():
    """Create vision tool function"""
    return setup_vision_tool()

def test_vision_api_connection(vision_tool):
    """Test basic API connectivity with a simple image"""
    # Use a known test image
    test_image = Path("./tests/fixtures/data/post_docs/post_0001.pdf")
    
    assert test_image.exists(), "Test image not found"
    
    # Make API call
    result = vision_tool(test_image, EXTRACT_PROMPT)
    
    # Basic validation
    assert isinstance(result, VisionTool)
    assert result.error_message is None
    assert result.tokens_used > 0
    assert result.processing_time > 0
    assert result.response is not None
    assert result.markdown is not None
    
    # Print response for inspection
    print("\nAPI Response:")
    print("-------------")
    print(f"Raw response length: {len(result.response)}")
    print(f"Markdown length: {len(result.markdown)}")
    print(f"Tokens used: {result.tokens_used}")
    print(f"Processing time: {result.processing_time:.2f}s")
    print("\nFirst 500 chars of markdown:")
    print(result.markdown[:500])

def test_vision_api_error_handling(vision_tool):
    """Test API error handling"""
    # Test with nonexistent file
    nonexistent = Path("nonexistent.pdf")
    result = vision_tool(nonexistent, EXTRACT_PROMPT)
    
    assert result.error_message is not None
    assert "not found" in result.error_message.lower()

def test_vision_api_with_invalid_prompt(vision_tool):
    """Test API with empty/invalid prompt"""
    test_image = Path("./tests/fixtures/data/post_docs/post_0001.pdf")
    
    # Test with empty prompt
    result = vision_tool(test_image, "")
    assert result.error_message is not None
    assert "prompt" in result.error_message.lower() 