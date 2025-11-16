import { useState, useEffect } from 'react'
import './ProductSelector.css'

function ProductSelector({ isConnected, onCollectionChange }) {
  const [collections, setCollections] = useState([])
  const [currentCollection, setCurrentCollection] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isOpen, setIsOpen] = useState(false)

  useEffect(() => {
    if (isConnected) {
      fetchCollections()
    }
  }, [isConnected])

  const fetchCollections = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/collections')
      if (!response.ok) {
        throw new Error('無法獲取產品列表')
      }
      const data = await response.json()
      setCollections(data.collections || [])
      setCurrentCollection(data.current_collection || '')
    } catch (error) {
      console.error('獲取產品列表失敗:', error)
    }
  }

  const handleSwitchCollection = async (collectionName) => {
    if (collectionName === currentCollection) {
      setIsOpen(false)
      return
    }

    setIsLoading(true)
    try {
      const response = await fetch('http://localhost:8000/api/switch-collection', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ collection_name: collectionName }),
      })

      if (!response.ok) {
        throw new Error('切換產品失敗')
      }

      const data = await response.json()
      setCurrentCollection(data.current_collection)
      setIsOpen(false)
      
      // 通知父組件
      if (onCollectionChange) {
        onCollectionChange(data.current_collection, data.document_count)
      }

      // 重新獲取列表以更新數量
      await fetchCollections()
    } catch (error) {
      console.error('切換產品失敗:', error)
      alert('❌ 切換產品失敗：' + error.message)
    } finally {
      setIsLoading(false)
    }
  }

  // 產品名稱顯示映射
  const getDisplayName = (name) => {
    const nameMap = {
      'airpods_manual': 'AirPods 使用手冊',
      'iphone_manual': 'iPhone 使用手冊',
      'ipad_manual': 'iPad 使用手冊',
      'macbook_manual': 'MacBook 使用手冊',
      'apple_watch_manual': 'Apple Watch 使用手冊',
    }
    return nameMap[name] || name
  }

  if (!isConnected || collections.length === 0) {
    return null
  }

  return (
    <div className="product-selector">
      <button 
        className="product-selector-button"
        onClick={() => setIsOpen(!isOpen)}
        disabled={isLoading}
      >
        <span className="product-icon">📚</span>
        <span className="product-name">{getDisplayName(currentCollection)}</span>
        <span className={`dropdown-arrow ${isOpen ? 'open' : ''}`}>▼</span>
      </button>

      {isOpen && (
        <div className="product-dropdown">
          <div className="dropdown-header">選擇產品知識庫</div>
          <div className="dropdown-list">
            {collections.map((collection) => (
              <button
                key={collection.name}
                className={`dropdown-item ${collection.name === currentCollection ? 'active' : ''}`}
                onClick={() => handleSwitchCollection(collection.name)}
                disabled={isLoading}
              >
                <span className="collection-name">
                  {getDisplayName(collection.name)}
                </span>
                <span className="collection-count">
                  {collection.count} 條
                </span>
                {collection.name === currentCollection && (
                  <span className="check-mark">✓</span>
                )}
              </button>
            ))}
          </div>
        </div>
      )}

      {isOpen && (
        <div 
          className="dropdown-backdrop" 
          onClick={() => setIsOpen(false)}
        />
      )}
    </div>
  )
}

export default ProductSelector
