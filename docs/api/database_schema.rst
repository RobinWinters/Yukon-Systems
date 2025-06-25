Database Schema
==============

The Database Schema module defines models and functions for session management.

.. automodule:: database_schema
   :members:
   :undoc-members:
   :show-inheritance:

Schema Structure
--------------

Sessions Table
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Column
     - Type
     - Description
   * - session_id
     - UUID
     - Primary key, unique identifier for the session
   * - user_id
     - String
     - Identifier for the user who owns the session
   * - start_time
     - DateTime
     - When the session was created
   * - end_time
     - DateTime
     - When the session was closed (null if active)
   * - metadata
     - JSON
     - Additional session information

Contexts Table
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Column
     - Type
     - Description
   * - context_id
     - UUID
     - Primary key, unique identifier for the context
   * - session_id
     - UUID
     - Foreign key reference to Sessions table
   * - content
     - Text
     - The context content
   * - type
     - String
     - Type of context (e.g., "system", "user", "file")
   * - created_at
     - DateTime
     - When the context was created

Conversations Table
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Column
     - Type
     - Description
   * - conversation_id
     - UUID
     - Primary key, unique identifier for the conversation
   * - session_id
     - UUID
     - Foreign key reference to Sessions table
   * - message_role
     - String
     - Role of the message sender (e.g., "user", "assistant")
   * - content
     - Text
     - The message content
   * - timestamp
     - DateTime
     - When the message was created
   * - metadata
     - JSON
     - Additional message information

Usage Examples
------------

Creating a new session:

.. code-block:: python

    from database_schema import create_session, get_session, add_context
    
    # Create a new session
    session_id = create_session(user_id="user123")
    
    # Add context to the session
    context_id = add_context(
        session_id=session_id,
        content="You are a helpful assistant.",
        type="system"
    )
    
    # Retrieve session information
    session = get_session(session_id)

