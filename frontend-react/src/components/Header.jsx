import { Home, Settings } from 'lucide-react'
import ProductSelector from './ProductSelector'
import './Header.css'

function Header({ 
  isConnected, 
  dbCount, 
  onSettingsClick, 
  onHomeClick,
  onCollectionChange 
}) {
  return (
    <div className="header">
      <div className="header-content">
        <button 
          className="home-button" 
          onClick={onHomeClick}
          title="回到主頁"
          aria-label="回到主頁"
        >
          <Home size={20} />
        </button>
        
        <h1>🎧 AirPods 智慧助手</h1>
        
        <ProductSelector 
          isConnected={isConnected}
          onCollectionChange={onCollectionChange}
        />
        
        <div className="settings-group">
          <button 
            className="settings-button" 
            onClick={onSettingsClick}
            title="設定 API Key"
            aria-label="設定"
          >
            <Settings size={20} />
          </button>
          <span className="settings-label">API Key 設定</span>
        </div>
        
        <div className="status-bar">
          <span className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
            {isConnected ? '🟢' : '🔴'}
          </span>
          <span className="status-text">
            {isConnected ? `已連接 (${dbCount} 筆資料)` : '連接失敗'}
          </span>
        </div>
      </div>
    </div>
  )
}

export default Header
