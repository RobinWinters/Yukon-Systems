Installation and Setup
====================

Requirements
-----------

Eve API requires:

* Python 3.8 or higher
* SQLAlchemy
* aiohttp
* Click
* Rich (for CLI formatting)

Basic Installation
----------------

You can install Eve API using pip:

.. code-block:: bash

    pip install eve-api

From Source
----------

To install from source:

.. code-block:: bash

    git clone https://github.com/yourusername/eve-api.git
    cd eve-api
    pip install -e .

Configuration
-----------

After installation, you'll need to configure your AI providers. Create a configuration file at ``~/.config/eve-api/config.json``:

.. code-block:: json

    {
        "providers": {
            "openai": {
                "api_key": "your-openai-api-key",
                "default_model": "gpt-4"
            },
            "anthropic": {
                "api_key": "your-anthropic-api-key",
                "default_model": "claude-2"
            },
            "local": {
                "url": "http://localhost:11434",
                "default_model": "llama2"
            }
        },
        "default_provider": "openai"
    }

Verify Installation
-----------------

To verify your installation, run:

.. code-block:: bash

    eve status

This should display information about your installation and configuration.

Database Setup
------------

Eve API uses SQLite by default for storing sessions, contexts, and conversations. The database will be automatically 
created at ``~/.local/share/eve-api/database.sqlite`` when you first run a command that requires database access.

If you want to use a different database location, set the ``EVE_DB_PATH`` environment variable:

.. code-block:: bash

    export EVE_DB_PATH=/path/to/your/database.sqlite

