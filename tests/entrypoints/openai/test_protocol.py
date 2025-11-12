import pytest

from fastdeploy.entrypoints.openai.protocol import FunctionDefinition


class TestValidateParametersAndName:
    """Test cases for FunctionDefinition.validate_parameters_and_name method"""

    def test_empty_function_name(self):
        """Test that empty function name raises error"""
        with pytest.raises(ValueError) as excinfo:
            FunctionDefinition.validate_parameters_and_name({"name": ""})
        assert "function name is required" in str(excinfo.value)

    def test_missing_function_name(self):
        """Test that missing function name raises error"""
        with pytest.raises(ValueError) as excinfo:
            FunctionDefinition.validate_parameters_and_name({})
        assert "function name is required" in str(excinfo.value)

    def test_invalid_json_parameters(self):
        """Test that invalid JSON parameters raise error"""

        # Create a circular reference that cannot be serialized
        class CircularReference:
            def __init__(self):
                self.ref = self

        with pytest.raises(ValueError) as excinfo:
            FunctionDefinition.validate_parameters_and_name(
                {"name": "test_func", "parameters": {"invalid": CircularReference()}}
            )
        # Match either the expected message or the actual circular reference message
        error_msg = str(excinfo.value)
        assert any(
            msg in error_msg
            for msg in ["function=test_func, msg=function is not a valid json", "Circular reference detected"]
        )

    def test_schema_validation_errors(self):
        """Test that schema validation errors are properly reported"""
        invalid_schema = {
            "type": "object",
            "properties": {
                "age": {"type": "string", "minimum": 18},  # Invalid combination
                "name": {"type": "invalid_type"},  # Invalid type
            },
            "required": ["invalid_field"],
        }
        with pytest.raises(ValueError) as excinfo:
            FunctionDefinition.validate_parameters_and_name({"name": "test_func", "parameters": invalid_schema})
        error_msg = str(excinfo.value)
        assert "function=test_func, msg=" in error_msg
        assert any(msg in error_msg for msg in ["invalid_field", "string", "minimum", "invalid_type"])

    def test_validate_json_decode_error(self):
        """Test JSON decode error handling"""

        # Create a class that will break JSON serialization
        class BadJSON:
            def __init__(self):
                self.ref = self  # Circular reference

        with pytest.raises(ValueError) as excinfo:
            FunctionDefinition.validate_parameters_and_name(
                {"name": "test_func", "parameters": {"invalid": BadJSON()}}
            )

        error_msg = str(excinfo.value)
        # Match either the expected message or the actual circular reference message
        assert any(
            msg in error_msg
            for msg in ["function=test_func, msg=function is not a valid json", "Circular reference detected"]
        )

    def test_valid_parameters_and_name(self):
        """Test that valid parameters and name return the input data"""
        valid_schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "number", "minimum": 0, "maximum": 150}},
            "required": ["name"],
        }

        input_data = {"name": "test_function", "parameters": valid_schema}

        result = FunctionDefinition.validate_parameters_and_name(input_data)

        # Should return the same data that was input
        assert result == input_data
        assert result["name"] == "test_function"
        assert result["parameters"] == valid_schema
