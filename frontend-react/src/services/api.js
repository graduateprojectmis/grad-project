import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// API 健康檢查
export const checkApiHealth = async () => {
  const response = await api.get('/api/health')
  return response.data
}

// 檢查 API Key 狀態
export const checkApiKeyStatus = async () => {
  const response = await api.get('/api/admin/api-key/status')
  return response.data
}

// 儲存 API Key
export const saveApiKey = async (apiKey, adminToken = '') => {
  const headers = adminToken ? { 'X-Admin-Token': adminToken } : {}
  const response = await api.post(
    '/api/admin/api-key',
    { api_key: apiKey },
    { headers }
  )
  return response.data
}

// 刪除 API Key
export const deleteApiKey = async (adminToken = '') => {
  const headers = adminToken ? { 'X-Admin-Token': adminToken } : {}
  const response = await api.delete('/api/admin/api-key', { headers })
  return response.data
}

// 發送問題
export const askQuestion = async (question, topK = 1) => {
  const response = await api.post('/api/ask', {
    question,
    top_k: topK,
  })
  return response.data
}

// 語義搜尋
export const searchDocuments = async (query, nResults = 5) => {
  const response = await api.post('/api/search', {
    query,
    n_results: nResults,
  })
  return response.data
}

// 上傳圖片
export const uploadImage = async (file) => {
  const formData = new FormData()
  formData.append('file', file)
  
  const response = await api.post('/api/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data
}

// 標註圖片
export const annotateImage = async (file, targetItem = 'objects') => {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('target_item', targetItem)
  
  const response = await api.post('/api/annotate-image', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data
}

export default api
