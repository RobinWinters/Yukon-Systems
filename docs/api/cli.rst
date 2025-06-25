Command Line Interface
=====================

The CLI module provides a comprehensive command-line interface for interacting with the system.

.. automodule:: cli
   :members:
   :undoc-members:
   :show-inheritance:

Available Commands
---------------

Pattern Management
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Command
     - Description
   * - ``eve pattern list``
     - List all available patterns
   * - ``eve pattern add <file>``
     - Add a new pattern from file
   * - ``eve pattern delete <id>``
     - Delete a pattern by ID
   * - ``eve pattern use <id> [--var name=value]``
     - Use a pattern with variables

Session Management
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Command
     - Description
   * - ``eve session list``
     - List all sessions
   * - ``eve session create``
     - Create a new session
   * - ``eve session current``
     - Show current session
   * - ``eve session switch <id>``
     - Switch to a different session
   * - ``eve session close [<id>]``
     - Close specified session or current session
   * - ``eve session context list``
     - List contexts for current session
   * - ``eve session context add <content> [--type type]``
     - Add context to current session

Model Management
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Command
     - Description
   * - ``eve model list``
     - List configured models
   * - ``eve model configure <provider> --key <key>``
     - Configure a model provider
   * - ``eve model set-default <provider>``
     - Set default model provider
   * - ``eve model chat [--provider provider] [--model model]``
     - Start interactive chat

System Commands
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Command
     - Description
   * - ``eve cleanup``
     - Clean up temporary resources
   * - ``eve status``
     - Show system status
   * - ``eve version``
     - Show version information

Usage Examples
------------

Basic usage:

.. code-block:: bash

    # Configure OpenAI provider
    eve model configure openai --key "your-api-key" --default-model "gpt-4"
    
    # Create a new session
    eve session create
    
    # Add system context
    eve session context add "You are a helpful assistant" --type system
    
    # Start chat
    eve model chat

