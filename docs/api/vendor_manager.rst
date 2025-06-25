Vendor Manager
=============

The Vendor Manager module handles integration with different AI model providers.

.. automodule:: vendor_manager
   :members:
   :undoc-members:
   :show-inheritance:

Provider Interface
---------------

Each provider implements the following interface:

.. code-block:: python

    class BaseVendorProvider(ABC):
        @abstractmethod
        async def generate(self, prompt: str, **kwargs) -> str:
            """Generate a response from the model."""
            pass
            
        @abstractmethod
        async def generate_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
            """Generate a streaming response from the model."""
            pass
            
        @property
        @abstractmethod
        def models(self) -> List[str]:
            """List available models."""
            pass

Supported Providers
----------------

The following providers are supported:

- **OpenAI** - Integration with OpenAI's API (GPT models)
- **Anthropic** - Integration with Anthropic's API (Claude models)
- **Local** - Integration with local models via Ollama

Usage Examples
------------

Basic usage:

.. code-block:: python

    import asyncio
    from vendor_manager import VendorManager
    
    async def main():
        # Initialize vendor manager
        manager = VendorManager()
        
        # Load configuration
        await manager.load_configuration()
        
        # Generate response
        response = await manager.generate(
            "What is the capital of France?",
            provider="openai",
            model="gpt-4"
        )
        
        print(response)
        
        # Streaming example
        async for chunk in manager.generate_stream(
            "Tell me a story about a dragon",
            provider="anthropic"
        ):
            print(chunk, end="", flush=True)
        
    asyncio.run(main())

