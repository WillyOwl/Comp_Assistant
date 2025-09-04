// Production Configuration for AI Composition Assistant Web Interface
// Copy this to web/config.js for production deployment

const CONFIG = {
    // API Configuration
    API: {
        // For local development
        DEV_URL: 'http://localhost:8000',
        
        // For deployment on willyzuo.net - UPDATE THIS URL IF NEEDED
        PROD_URL: 'https://willyzuo.net/composition-assistant/api',
        
        // Authentication token - CHANGE THIS TO A SECURE TOKEN IN PRODUCTION!
        AUTH_TOKEN: 'secure-production-token-2024',
        
        // Request timeout (in milliseconds)
        TIMEOUT: 30000
    },
    
    // UI Configuration
    UI: {
        MAX_FILE_SIZE: 50 * 1024 * 1024, // 50MB
        SUPPORTED_FORMATS: ['image/jpeg', 'image/jpg', 'image/png', 'image/tiff', 'image/bmp', 'image/webp'],
        
        // Animation settings
        ANIMATION_DURATION: 300,
        
        // Default analysis configuration
        DEFAULT_CONFIG: {
            analysis_depth: 'standard',
            max_suggestions: 5,
            include_technical_metrics: true,
            return_visualizations: false
        }
    },
    
    // Feature flags
    FEATURES: {
        ENABLE_BATCH_PROCESSING: false,
        ENABLE_ADVANCED_CONFIG: true,
        ENABLE_RESULT_EXPORT: true,
        ENABLE_COMPARISON_MODE: false
    },
    
    // Analytics (optional)
    ANALYTICS: {
        TRACK_USAGE: true,  // Enable in production
        GOOGLE_ANALYTICS_ID: '', // Add your GA ID if needed
    }
};

// Auto-detect environment and set API URL
CONFIG.API.BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? CONFIG.API.DEV_URL 
    : CONFIG.API.PROD_URL;

// Export configuration
window.APP_CONFIG = CONFIG;
