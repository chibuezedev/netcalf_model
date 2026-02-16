from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import tensorflow as tf
import json
from datetime import datetime
import logging


MODEL_PATH = "./outputs/production_model.h5"
PREPROCESSOR_PATH = "./outputs/preprocessors/preprocessor.pkl"
METADATA_PATH = "./outputs/model_metadata.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Network IDS API",
    description="Real-time Network Intrusion Detection System using Deep Learning",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class NetworkTraffic(BaseModel):
    """Network traffic feature vector"""

    features: List[float] = Field(
        ..., description="Network flow features (80+ features from CICFlowMeter)"
    )

    class Config:
        schema_extra = {"example": {"features": [0.5] * 80}}


class PredictionResponse(BaseModel):
    """Prediction response model"""

    success: bool
    prediction: str
    prediction_id: int
    confidence: float
    all_probabilities: Dict[str, float]
    action: str
    timestamp: str
    processing_time_ms: float


class BatchNetworkTraffic(BaseModel):
    """Batch prediction request"""

    traffic_batch: List[List[float]] = Field(
        ..., description="List of network flow feature vectors"
    )


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""

    success: bool
    predictions: List[Dict]
    total_samples: int
    processing_time_ms: float
    timestamp: str


class ModelStatus(BaseModel):
    """Model status response"""

    status: str
    model_loaded: bool
    model_name: str
    num_classes: int
    class_names: List[str]
    input_features: int
    model_metrics: Optional[Dict]


class IDSModelServer:
    """Model server for IDS inference"""

    def __init__(self, model_path: str, preprocessor_path: str, metadata_path: str):
        """
        Initialize model server

        Args:
            model_path: Path to trained model (.h5)
            preprocessor_path: Path to preprocessor (.pkl)
            metadata_path: Path to metadata (.json)
        """
        self.model = None
        self.preprocessor = None
        self.metadata = None
        self.class_names = []
        self.model_name = "Unknown"

        self._load_model(model_path)
        self._load_preprocessor(preprocessor_path)
        self._load_metadata(metadata_path)

        logger.info("✓ Model server initialized successfully")

    def _load_model(self, model_path: str):
        """Load trained Keras model"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"✓ Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"✗ Failed to load model: {str(e)}")
            raise

    def _load_preprocessor(self, preprocessor_path: str):
        """Load preprocessor"""
        try:
            from preprocessors.preprocessor import AdvancedPreprocessor

            self.preprocessor = AdvancedPreprocessor.load(preprocessor_path)
            logger.info(f"✓ Preprocessor loaded from {preprocessor_path}")
        except Exception as e:
            logger.error(f"✗ Failed to load preprocessor: {str(e)}")
            raise

    def _load_metadata(self, metadata_path: str):
        """Load model metadata"""
        try:
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
            self.class_names = self.metadata.get("class_names", [])
            self.model_name = self.metadata.get("model_name", "Unknown")
            logger.info(f"✓ Metadata loaded: {self.model_name}")
        except Exception as e:
            logger.warning(f"⚠ Failed to load metadata: {str(e)}")
            self.class_names = [
                "Benign",
                "Analysis",
                "Backdoor",
                "DoS",
                "Exploits",
                "Fuzzers",
                "Generic",
                "Reconnaissance",
                "Shellcode",
                "Worms",
            ]

    def preprocess_features(self, features: np.ndarray) -> np.ndarray:
        """
        Preprocess raw features

        Args:
            features: Raw feature array

        Returns:
            Preprocessed features
        """
        features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)

        features_processed = self.preprocessor.transform(features)

        return features_processed

    def predict(self, features: np.ndarray) -> Dict:
        """
        Make prediction on network traffic

        Args:
            features: Preprocessed feature array

        Returns:
            Prediction dictionary
        """
        start_time = datetime.now()

        if len(self.model.input_shape) == 3:
            features = features.reshape(features.shape[0], features.shape[1], 1)

        probabilities = self.model.predict(features, verbose=0)

        prediction_id = int(np.argmax(probabilities[0]))
        confidence = float(probabilities[0][prediction_id])
        prediction_name = self.class_names[prediction_id]

        action = "ALLOW" if prediction_id == 0 else "BLOCK"

        all_probs = {
            self.class_names[i]: float(probabilities[0][i])
            for i in range(len(self.class_names))
        }

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return {
            "prediction": prediction_name,
            "prediction_id": prediction_id,
            "confidence": confidence,
            "all_probabilities": all_probs,
            "action": action,
            "processing_time_ms": processing_time,
        }

    def predict_batch(self, features_batch: np.ndarray) -> List[Dict]:
        """
        Batch prediction

        Args:
            features_batch: Batch of preprocessed features

        Returns:
            List of predictions
        """
        start_time = datetime.now()

        if len(self.model.input_shape) == 3:
            features_batch = features_batch.reshape(
                features_batch.shape[0], features_batch.shape[1], 1
            )

        probabilities = self.model.predict(features_batch, verbose=0)

        predictions = []
        for i, probs in enumerate(probabilities):
            prediction_id = int(np.argmax(probs))
            confidence = float(probs[prediction_id])
            prediction_name = self.class_names[prediction_id]
            action = "ALLOW" if prediction_id == 0 else "BLOCK"

            predictions.append(
                {
                    "sample_id": i,
                    "prediction": prediction_name,
                    "prediction_id": prediction_id,
                    "confidence": confidence,
                    "action": action,
                    "all_probabilities": {
                        self.class_names[j]: float(probs[j])
                        for j in range(len(self.class_names))
                    },
                }
            )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        for pred in predictions:
            pred["processing_time_ms"] = processing_time / len(predictions)

        return predictions

    def get_status(self) -> Dict:
        """Get model status"""
        return {
            "status": "active",
            "model_loaded": self.model is not None,
            "model_name": self.model_name,
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "input_features": self.model.input_shape[1] if self.model else 0,
            "model_metrics": self.metadata.get("metrics", {}) if self.metadata else {},
        }


try:
    model_server = IDSModelServer(MODEL_PATH, PREPROCESSOR_PATH, METADATA_PATH)
    logger.info("✓ Model server initialized")
except Exception as e:
    logger.error(f"✗ Failed to initialize model server: {str(e)}")
    model_server = None


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint"""
    return {
        "message": "Network IDS API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
        },
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    if model_server is None:
        raise HTTPException(status_code=503, detail="Model server not initialized")

    return {
        "status": "healthy",
        "model_loaded": model_server.model is not None,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/status", response_model=ModelStatus, tags=["Model"])
async def get_model_status():
    """Get detailed model status"""
    if model_server is None:
        raise HTTPException(status_code=503, detail="Model server not initialized")

    return model_server.get_status()


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_traffic(traffic: NetworkTraffic):
    """
    Predict network traffic classification

    Real-time classification of network flow into benign or attack categories.
    """
    if model_server is None:
        raise HTTPException(status_code=503, detail="Model server not initialized")

    try:
        features = np.array([traffic.features])

        features_processed = model_server.preprocess_features(features)
        result = model_server.predict(features_processed)

        response = PredictionResponse(
            success=True,
            prediction=result["prediction"],
            prediction_id=result["prediction_id"],
            confidence=result["confidence"],
            all_probabilities=result["all_probabilities"],
            action=result["action"],
            timestamp=datetime.now().isoformat(),
            processing_time_ms=result["processing_time_ms"],
        )

        return response

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(batch_data: BatchNetworkTraffic):
    """
    Batch prediction for multiple network flows

    Efficient batch processing for high-throughput scenarios.
    """
    if model_server is None:
        raise HTTPException(status_code=503, detail="Model server not initialized")

    try:
        start_time = datetime.now()

        features_batch = np.array(batch_data.traffic_batch)

        features_processed = model_server.preprocess_features(features_batch)

        predictions = model_server.predict_batch(features_processed)

        total_time = (datetime.now() - start_time).total_seconds() * 1000

        response = BatchPredictionResponse(
            success=True,
            predictions=predictions,
            total_samples=len(predictions),
            processing_time_ms=total_time,
            timestamp=datetime.now().isoformat(),
        )

        return response

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/classes", tags=["Model"])
async def get_classes():
    """Get all attack classes"""
    if model_server is None:
        raise HTTPException(status_code=503, detail="Model server not initialized")

    return {
        "classes": model_server.class_names,
        "num_classes": len(model_server.class_names),
        "description": {
            "Benign": "Normal network traffic",
            "Analysis": "Traffic analysis attacks",
            "Backdoor": "Backdoor access attempts",
            "DoS": "Denial of Service attacks",
            "Exploits": "Vulnerability exploitation",
            "Fuzzers": "Fuzzing attacks",
            "Generic": "Generic cryptographic attacks",
            "Reconnaissance": "Network reconnaissance",
            "Shellcode": "Shellcode injection",
            "Worms": "Worm propagation",
        },
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc),
            "timestamp": datetime.now().isoformat(),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_server:app", host="0.0.0.0", port=8000, reload=False, log_level="info"
    )
