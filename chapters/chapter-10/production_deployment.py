#!/usr/bin/env python3
"""
Chapter 10: Production Deployment with FastAPI

This script demonstrates production-ready deployment using FastAPI,
with proper error handling, monitoring, and scalability features.

Pocket Agents: A Practical Guide to On‑Device Artificial Intelligence
"""

import time
import asyncio
import psutil
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

# FastAPI and related imports
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
    from llama_cpp import Llama
    PRODUCTION_AVAILABLE = True
except ImportError as e:
    PRODUCTION_AVAILABLE = False
    print(f"⚠️ Production frameworks not available: {e}")
    print("Install with: pip install fastapi uvicorn llama-cpp-python")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenerationRequest(BaseModel):
    """Request model for text generation"""
    prompt: str = Field(..., description="The input prompt for text generation")
    max_tokens: int = Field(100, ge=1, le=1000, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    stream: bool = Field(False, description="Whether to stream the response")

class GenerationResponse(BaseModel):
    """Response model for text generation"""
    text: str = Field(..., description="Generated text")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    inference_time: float = Field(..., description="Inference time in seconds")
    throughput: float = Field(..., description="Tokens per second")
    model_info: Dict[str, Any] = Field(..., description="Model information")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    memory_usage: float
    uptime: float
    total_requests: int

class ProductionAIEngine:
    """Production-ready AI engine with monitoring and error handling"""
    
    def __init__(self, model_path: str = None, **kwargs):
        """Initialize production AI engine"""
        
        if not PRODUCTION_AVAILABLE:
            raise ImportError("Production frameworks are required")
        
        self.model_path = model_path
        self.model = None
        self.model_loaded = False
        self.start_time = time.time()
        self.total_requests = 0
        self.error_count = 0
        
        # Performance tracking
        self.inference_times = []
        self.token_counts = []
        self.memory_usage = []
        
        # Default parameters for production
        default_params = {
            'n_ctx': 2048,
            'n_batch': 512,
            'n_threads': 4,
            'n_gpu_layers': 0,
            'use_mmap': True,
            'use_mlock': False,
            'verbose': False
        }
        
        self.params = {**default_params, **kwargs}
        
        logger.info("Initializing production AI engine...")
        self._load_model()
    
    def _load_model(self):
        """Load the model with error handling"""
        try:
            if self.model_path and Path(self.model_path).exists():
                logger.info(f"Loading model from {self.model_path}")
                self.model = Llama(model_path=self.model_path, **self.params)
                self.model_loaded = True
                logger.info("✅ Model loaded successfully")
            else:
                logger.warning("No model path provided or model not found")
                logger.info("Using mock responses for demonstration")
                self.model_loaded = False
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model_loaded = False
    
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text with comprehensive error handling and monitoring"""
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            self.total_requests += 1
            
            if not self.model_loaded:
                # Mock response for demonstration
                await asyncio.sleep(0.1)  # Simulate processing time
                generated_text = f"Mock response to: '{request.prompt[:50]}...' (Model not loaded - this is a demo response)"
                tokens_generated = len(generated_text.split())
            else:
                # Real model inference
                response = self.model(
                    request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    stop=request.stop or ["</s>", "\n\n"],
                    echo=False
                )
                
                generated_text = response['choices'][0]['text']
                tokens_generated = len(generated_text.split())
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Calculate metrics
            inference_time = end_time - start_time
            throughput = tokens_generated / inference_time if inference_time > 0 else 0
            memory_used = end_memory - start_memory
            
            # Track performance
            self.inference_times.append(inference_time)
            self.token_counts.append(tokens_generated)
            self.memory_usage.append(memory_used)
            
            # Keep only last 100 measurements
            if len(self.inference_times) > 100:
                self.inference_times = self.inference_times[-100:]
                self.token_counts = self.token_counts[-100:]
                self.memory_usage = self.memory_usage[-100:]
            
            return GenerationResponse(
                text=generated_text,
                tokens_generated=tokens_generated,
                inference_time=inference_time,
                throughput=throughput,
                model_info={
                    "model_loaded": self.model_loaded,
                    "model_path": str(self.model_path) if self.model_path else None,
                    "context_size": self.params['n_ctx'],
                    "batch_size": self.params['n_batch'],
                    "threads": self.params['n_threads']
                }
            )
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    
    def get_health_status(self) -> HealthResponse:
        """Get health status of the engine"""
        
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        uptime = time.time() - self.start_time
        
        return HealthResponse(
            status="healthy" if self.model_loaded else "degraded",
            model_loaded=self.model_loaded,
            memory_usage=current_memory,
            uptime=uptime,
            total_requests=self.total_requests
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        
        if not self.inference_times:
            return {"message": "No performance data available"}
        
        avg_inference_time = sum(self.inference_times) / len(self.inference_times)
        avg_tokens = sum(self.token_counts) / len(self.token_counts)
        avg_memory = sum(self.memory_usage) / len(self.memory_usage)
        avg_throughput = avg_tokens / avg_inference_time if avg_inference_time > 0 else 0
        
        return {
            "average_inference_time": avg_inference_time,
            "average_tokens": avg_tokens,
            "average_memory_usage": avg_memory,
            "average_throughput": avg_throughput,
            "total_requests": self.total_requests,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.total_requests, 1),
            "measurements_count": len(self.inference_times)
        }

# Global engine instance
engine = None

def create_production_app() -> FastAPI:
    """Create production FastAPI application"""
    
    app = FastAPI(
        title="On-Device AI API",
        description="Production-ready on-device AI inference API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize the AI engine on startup"""
        global engine
        logger.info("Starting production AI API...")
        
        # Initialize engine (use demo model if none provided)
        model_path = "models/llama-3.1-7b-instruct-q4_k_m.gguf"  # Update this path
        engine = ProductionAIEngine(model_path=model_path)
        
        logger.info("Production AI API started successfully")
    
    @app.post("/generate", response_model=GenerationResponse)
    async def generate_text(request: GenerationRequest):
        """Generate text using the AI engine"""
        
        if engine is None:
            raise HTTPException(status_code=503, detail="AI engine not initialized")
        
        try:
            response = await engine.generate(request)
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in generation: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint"""
        
        if engine is None:
            raise HTTPException(status_code=503, detail="AI engine not initialized")
        
        return engine.get_health_status()
    
    @app.get("/stats")
    async def get_stats():
        """Get performance statistics"""
        
        if engine is None:
            raise HTTPException(status_code=503, detail="AI engine not initialized")
        
        return engine.get_performance_stats()
    
    @app.get("/")
    async def root():
        """Root endpoint with API information"""
        return {
            "message": "On-Device AI API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
            "stats": "/stats"
        }
    
    return app

def demonstrate_production_deployment():
    """Demonstrate production deployment capabilities"""
    
    print("=" * 70)
    print("Chapter 10: Production Deployment Demo")
    print("Pocket Agents: A Practical Guide to On‑Device Artificial Intelligence")
    print("=" * 70)
    
    if not PRODUCTION_AVAILABLE:
        print("❌ Production frameworks not available")
        print("💡 Install with: pip install fastapi uvicorn llama-cpp-python")
        return
    
    try:
        # Test engine initialization
        print("🧪 Testing production deployment...")
        engine = ProductionAIEngine()
        
        # Test health status
        health = engine.get_health_status()
        print(f"📊 Health Status:")
        print(f"   Status: {health.status}")
        print(f"   Model Loaded: {health.model_loaded}")
        print(f"   Memory Usage: {health.memory_usage:.1f} MB")
        print(f"   Uptime: {health.uptime:.1f}s")
        
        # Test generation
        if health.model_loaded or True:  # Test even with mock
            print(f"\n🧪 Testing generation...")
            request = GenerationRequest(
                prompt="Hello! How are you?",
                max_tokens=50,
                temperature=0.7
            )
            
            response = asyncio.run(engine.generate(request))
            print(f"📝 Generated: {response.text}")
            print(f"⏱️ Inference time: {response.inference_time:.3f}s")
            print(f"🚀 Throughput: {response.throughput:.1f} tokens/sec")
        
        # Show performance stats
        stats = engine.get_performance_stats()
        if "message" not in stats:
            print(f"\n📊 Performance Statistics:")
            print(f"   Average inference time: {stats['average_inference_time']:.3f}s")
            print(f"   Average throughput: {stats['average_throughput']:.1f} tokens/sec")
            print(f"   Total requests: {stats['total_requests']}")
            print(f"   Error rate: {stats['error_rate']:.1%}")
        
        print(f"\n✅ Production deployment ready!")
        print(f"💡 Key features:")
        print(f"   • FastAPI with automatic documentation")
        print(f"   • Comprehensive error handling")
        print(f"   • Performance monitoring")
        print(f"   • Health checks and metrics")
        print(f"   • CORS support for web clients")
        print(f"   • Async/await for scalability")
        
    except Exception as e:
        print(f"❌ Error in production deployment demo: {e}")

def launch_production_server():
    """Launch production server"""
    
    print("🚀 Launching production AI API server...")
    print("=" * 70)
    
    if not PRODUCTION_AVAILABLE:
        print("❌ Production frameworks not available")
        return
    
    app = create_production_app()
    
    print("🌐 Starting FastAPI server...")
    print("   URL: http://localhost:8000")
    print("   Docs: http://localhost:8000/docs")
    print("   Health: http://localhost:8000/health")
    print("   Stats: http://localhost:8000/stats")
    print("\n💡 Press Ctrl+C to stop the server")
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            reload=False
        )
    except KeyboardInterrupt:
        print("\n🛑 Stopping production server...")
        print("✅ Server stopped")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        launch_production_server()
    else:
        demonstrate_production_deployment()
