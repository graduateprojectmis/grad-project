import { useState, useEffect } from 'react'
import { X, Eye, EyeOff, Lock, CheckCircle, XCircle } from 'lucide-react'
import { checkApiKeyStatus, saveApiKey, deleteApiKey } from '../services/api'
import './ApiKeyModal.css'

function ApiKeyModal({ onClose, onApiKeySaved }) {
  const [apiKey, setApiKey] = useState('')
  const [showApiKey, setShowApiKey] = useState(false)
  const [keyStatus, setKeyStatus] = useState({ exists: false, masked: '' })
  const [statusMessage, setStatusMessage] = useState('檢查中...')
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    checkStatus()
  }, [])

  const checkStatus = async () => {
    try {
      const data = await checkApiKeyStatus()
      setKeyStatus(data)
      setStatusMessage(
        data.exists 
          ? `已設定 API Key (${data.masked})` 
          : '未設定 API Key'
      )
    } catch (error) {
      setStatusMessage('無法檢查狀態')
    }
  }

  const handleSave = async () => {
    const key = apiKey.trim()
    
    if (!key) {
      alert('請輸入 API Key')
      return
    }

    if (!key.startsWith('sk-')) {
      alert('API Key 格式似乎不正確，應該以 "sk-" 開頭')
      return
    }

    try {
      setIsLoading(true)
      await saveApiKey(key)
      alert('API Key 已安全儲存到伺服器環境變數 (.env)。')
      setApiKey('')
      await checkStatus()
      onApiKeySaved()
      onClose()
    } catch (error) {
      console.error('儲存 API Key 錯誤:', error)
      alert('儲存失敗：' + (error.response?.data?.detail || error.message))
    } finally {
      setIsLoading(false)
    }
  }

  const handleClear = async () => {
    if (!confirm('確定要清除伺服器中的 API Key 嗎？')) {
      return
    }

    try {
      setIsLoading(true)
      await deleteApiKey()
      alert('伺服器中的 API Key 已清除')
      setApiKey('')
      await checkStatus()
      onApiKeySaved()
    } catch (error) {
      console.error('刪除 API Key 錯誤:', error)
      alert('刪除失敗：' + (error.response?.data?.detail || error.message))
    } finally {
      setIsLoading(false)
    }
  }

  const handleBackdropClick = (e) => {
    if (e.target === e.currentTarget) {
      onClose()
    }
  }

  return (
    <div className="modal-overlay" onClick={handleBackdropClick}>
      <div className="modal-content">
        <div className="modal-header">
          <h2>🔑 設定 OpenAI API Key</h2>
          <button 
            className="modal-close" 
            onClick={onClose}
            aria-label="關閉"
          >
            <X size={24} />
          </button>
        </div>
        
        <div className="modal-body">
          <div className={`api-key-status ${keyStatus.exists ? 'has-key' : ''}`}>
            <span className="status-icon">
              {keyStatus.exists ? <CheckCircle size={20} /> : <Lock size={20} />}
            </span>
            <span className="status-message">{statusMessage}</span>
          </div>
          
          <div className="form-group">
            <label htmlFor="apiKeyInput">OpenAI API Key</label>
            <div className="api-key-input-group">
              <input
                id="apiKeyInput"
                type={showApiKey ? 'text' : 'password'}
                placeholder="sk-..."
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                autoComplete="off"
              />
              <button
                type="button"
                className="toggle-visibility"
                onClick={() => setShowApiKey(!showApiKey)}
                title={showApiKey ? '隱藏' : '顯示'}
                aria-label={showApiKey ? '隱藏' : '顯示'}
              >
                {showApiKey ? <EyeOff size={18} /> : <Eye size={18} />}
              </button>
            </div>
            <small className="help-text">
              API Key 將安全儲存在伺服器環境變數（.env）
            </small>
          </div>
        </div>
        
        <div className="modal-footer">
          <button
            className="btn-secondary"
            onClick={handleClear}
            disabled={isLoading}
          >
            清除伺服器中的 Key
          </button>
          <button
            className="btn-primary"
            onClick={handleSave}
            disabled={isLoading}
          >
            {isLoading ? '處理中...' : '儲存到伺服器'}
          </button>
        </div>
      </div>
    </div>
  )
}

export default ApiKeyModal
