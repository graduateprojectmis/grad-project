// API 配置
const API_CONFIG = {
    // 開發環境
    DEV_URL: 'http://localhost:8000',
    
    // 生產環境（部署後修改這裡）
    PROD_URL: 'https://your-api-domain.com',
    
    // 自動偵測環境
    get BASE_URL() {
        return window.location.hostname === 'localhost' || 
               window.location.hostname === '127.0.0.1'
            ? this.DEV_URL 
            : this.PROD_URL;
    },
    
    // API 端點
    ENDPOINTS: {
        HEALTH: '/api/health',
        ASK: '/api/ask',
        SEARCH: '/api/search',
        UPLOAD: '/api/upload',
        ADMIN_KEY: '/api/admin/api-key',
        ADMIN_KEY_STATUS: '/api/admin/api-key/status'
    }
};

// 導出配置
window.API_CONFIG = API_CONFIG;
