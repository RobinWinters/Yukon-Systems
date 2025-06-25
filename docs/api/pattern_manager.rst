Pattern Manager
==============

The Pattern Manager module handles template-based interactions with AI models.

.. automodule:: pattern_manager
   :members:
   :undoc-members:
   :show-inheritance:

Pattern Schema
------------

Patterns are stored in the following format:

.. code-block:: json

    {
        "id": "greeting_pattern",
        "name": "Greeting",
        "description": "A simple greeting pattern",
        "template": "Hello, {{name}}! How can I assist you today?",
        "variables": [
            {
                "name": "name",
                "description": "The user's name",
                "required": true,
                "default": "User"
            }
        ],
        "metadata": {
            "category": "general",
            "tags": ["greeting", "welcome"]
        }
    }

Usage Examples
------------

Basic pattern retrieval:

.. code-block:: python

    from pattern_manager import PatternManager
    
    # Initialize pattern manager
    manager = PatternManager()
    
    # Load patterns from file
    manager.load_patterns_from_file('patterns.json')
    
    # Get a pattern by ID
    pattern = manager.get_pattern('greeting_pattern')
    
    # Render pattern with variables
    rendered = manager.render_pattern(pattern, {'name': 'Alice'})
    # Output: "Hello, Alice! How can I assist you today?"

