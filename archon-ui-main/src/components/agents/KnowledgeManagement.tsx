import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Input } from '../ui/input';
import { Progress } from '../ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { useToast } from '../../hooks/useToast';
import { 
  BookOpen,
  Search,
  Upload,
  Download,
  Brain,
  Database,
  FileText,
  Globe,
  Tag,
  Clock,
  Zap,
  Target,
  Filter,
  MoreHorizontal,
  Plus,
  Eye,
  Edit,
  Trash2,
  Share
} from 'lucide-react';

interface KnowledgeItem {
  id: string;
  title: string;
  type: 'document' | 'code' | 'web' | 'note';
  content_preview: string;
  tags: string[];
  created_at: string;
  updated_at: string;
  source: string;
  embedding_status: 'processing' | 'completed' | 'failed';
  usage_count: number;
  relevance_score?: number;
}

interface KnowledgeStats {
  total_items: number;
  documents: number;
  code_snippets: number;
  web_sources: number;
  total_embeddings: number;
  storage_used_mb: number;
  search_queries_today: number;
}

interface SearchResult {
  id: string;
  title: string;
  content: string;
  relevance_score: number;
  source: string;
  type: string;
}

interface KnowledgeManagementProps {
  agents: any[];
}

export const KnowledgeManagement: React.FC<KnowledgeManagementProps> = ({ agents }) => {
  const [knowledgeItems, setKnowledgeItems] = useState<KnowledgeItem[]>([]);
  const [stats, setStats] = useState<KnowledgeStats>({
    total_items: 0,
    documents: 0,
    code_snippets: 0,
    web_sources: 0,
    total_embeddings: 0,
    storage_used_mb: 0,
    search_queries_today: 0
  });
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [selectedType, setSelectedType] = useState<string>('all');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const { toast } = useToast();

  useEffect(() => {
    loadKnowledgeData();
  }, []);

  const loadKnowledgeData = async () => {
    try {
      // Simulate loading knowledge data
      const mockItems: KnowledgeItem[] = [
        {
          id: '1',
          title: 'React Best Practices Guide',
          type: 'document',
          content_preview: 'A comprehensive guide to React development best practices including hooks, performance optimization...',
          tags: ['react', 'javascript', 'frontend', 'best-practices'],
          created_at: '2024-01-15T10:30:00Z',
          updated_at: '2024-01-20T14:22:00Z',
          source: 'documentation/react-guide.md',
          embedding_status: 'completed',
          usage_count: 45
        },
        {
          id: '2',
          title: 'TypeScript Type Guards Implementation',
          type: 'code',
          content_preview: 'function isString(value: unknown): value is string { return typeof value === "string"; }',
          tags: ['typescript', 'type-guards', 'utility'],
          created_at: '2024-01-18T09:15:00Z',
          updated_at: '2024-01-18T09:15:00Z',
          source: 'src/utils/typeGuards.ts',
          embedding_status: 'completed',
          usage_count: 23
        },
        {
          id: '3',
          title: 'Modern Web Development Trends 2024',
          type: 'web',
          content_preview: 'Overview of the latest web development trends including AI integration, serverless architecture...',
          tags: ['trends', 'web-development', '2024', 'ai'],
          created_at: '2024-01-10T16:45:00Z',
          updated_at: '2024-01-10T16:45:00Z',
          source: 'https://webdev-trends.com/2024',
          embedding_status: 'completed',
          usage_count: 67
        }
      ];

      const mockStats: KnowledgeStats = {
        total_items: mockItems.length,
        documents: mockItems.filter(item => item.type === 'document').length,
        code_snippets: mockItems.filter(item => item.type === 'code').length,
        web_sources: mockItems.filter(item => item.type === 'web').length,
        total_embeddings: mockItems.filter(item => item.embedding_status === 'completed').length,
        storage_used_mb: 156.7,
        search_queries_today: 42
      };

      setKnowledgeItems(mockItems);
      setStats(mockStats);
    } catch (error) {
      console.error('Failed to load knowledge data:', error);
      toast({
        title: "Error",
        description: "Failed to load knowledge data",
        variant: "destructive"
      });
    }
  };

  const performSearch = async () => {
    if (!searchQuery.trim()) return;

    setIsSearching(true);
    try {
      // Simulate search with relevance scoring
      const mockResults: SearchResult[] = knowledgeItems
        .filter(item => 
          item.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
          item.content_preview.toLowerCase().includes(searchQuery.toLowerCase()) ||
          item.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
        )
        .map(item => ({
          id: item.id,
          title: item.title,
          content: item.content_preview,
          relevance_score: Math.random() * 0.4 + 0.6, // Mock score between 0.6-1.0
          source: item.source,
          type: item.type
        }))
        .sort((a, b) => b.relevance_score - a.relevance_score);

      setSearchResults(mockResults);
      
      toast({
        title: "Search Complete",
        description: `Found ${mockResults.length} relevant items`,
        variant: "success"
      });
    } catch (error) {
      toast({
        title: "Search Error",
        description: "Failed to perform search",
        variant: "destructive"
      });
    } finally {
      setIsSearching(false);
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files?.length) return;

    setIsUploading(true);
    setUploadProgress(0);

    try {
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        
        // Simulate upload progress
        for (let progress = 0; progress <= 100; progress += 10) {
          setUploadProgress(progress);
          await new Promise(resolve => setTimeout(resolve, 100));
        }

        // Add uploaded file to knowledge base
        const newItem: KnowledgeItem = {
          id: `upload_${Date.now()}_${i}`,
          title: file.name,
          type: file.type.includes('text') ? 'document' : 'document',
          content_preview: 'Uploaded file content preview...',
          tags: ['uploaded', 'new'],
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          source: `uploads/${file.name}`,
          embedding_status: 'processing',
          usage_count: 0
        };

        setKnowledgeItems(prev => [newItem, ...prev]);
      }

      toast({
        title: "Upload Complete",
        description: `Successfully uploaded ${files.length} file(s)`,
        variant: "success"
      });
    } catch (error) {
      toast({
        title: "Upload Error",
        description: "Failed to upload files",
        variant: "destructive"
      });
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'document':
        return <FileText className="w-4 h-4 text-blue-600" />;
      case 'code':
        return <Zap className="w-4 h-4 text-purple-600" />;
      case 'web':
        return <Globe className="w-4 h-4 text-green-600" />;
      case 'note':
        return <BookOpen className="w-4 h-4 text-orange-600" />;
      default:
        return <FileText className="w-4 h-4 text-gray-600" />;
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'document':
        return 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400';
      case 'code':
        return 'bg-purple-100 text-purple-800 dark:bg-purple-900/20 dark:text-purple-400';
      case 'web':
        return 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400';
      case 'note':
        return 'bg-orange-100 text-orange-800 dark:bg-orange-900/20 dark:text-orange-400';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <Database className="w-3 h-3 text-green-500" />;
      case 'processing':
        return <Clock className="w-3 h-3 text-yellow-500" />;
      case 'failed':
        return <Target className="w-3 h-3 text-red-500" />;
      default:
        return <Clock className="w-3 h-3 text-gray-500" />;
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  const filteredItems = knowledgeItems.filter(item => 
    selectedType === 'all' || item.type === selectedType
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <BookOpen className="w-8 h-8 text-blue-600" />
            Knowledge Management
          </h2>
          <p className="text-gray-600 dark:text-gray-400">
            Centralized knowledge base for AI agents and team collaboration
          </p>
        </div>
        <div className="flex gap-2">
          <input
            type="file"
            multiple
            accept=".txt,.md,.pdf,.docx"
            onChange={handleFileUpload}
            className="hidden"
            id="file-upload"
          />
          <label htmlFor="file-upload">
            <Button variant="outline" className="flex items-center gap-2" disabled={isUploading}>
              <Upload className="w-4 h-4" />
              Upload Files
            </Button>
          </label>
          <Button className="flex items-center gap-2">
            <Plus className="w-4 h-4" />
            Add Knowledge
          </Button>
        </div>
      </div>

      {/* Upload Progress */}
      {isUploading && (
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium">Uploading files...</span>
              <span className="text-sm text-gray-500">{uploadProgress}%</span>
            </div>
            <Progress value={uploadProgress} className="h-2" />
          </CardContent>
        </Card>
      )}

      {/* Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-blue-600">{stats.total_items}</div>
                <div className="text-sm text-gray-500">Total Items</div>
              </div>
              <BookOpen className="w-8 h-8 text-blue-400" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-green-600">{stats.total_embeddings}</div>
                <div className="text-sm text-gray-500">Embeddings</div>
              </div>
              <Brain className="w-8 h-8 text-green-400" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-purple-600">{stats.storage_used_mb.toFixed(1)}MB</div>
                <div className="text-sm text-gray-500">Storage</div>
              </div>
              <Database className="w-8 h-8 text-purple-400" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-orange-600">{stats.search_queries_today}</div>
                <div className="text-sm text-gray-500">Searches Today</div>
              </div>
              <Search className="w-8 h-8 text-orange-400" />
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="browse" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="browse">📚 Browse</TabsTrigger>
          <TabsTrigger value="search">🔍 Search</TabsTrigger>
          <TabsTrigger value="analytics">📊 Analytics</TabsTrigger>
        </TabsList>

        {/* Browse Tab */}
        <TabsContent value="browse" className="space-y-4">
          {/* Filters */}
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <Filter className="w-4 h-4 text-gray-500" />
                  <select
                    value={selectedType}
                    onChange={(e) => setSelectedType(e.target.value)}
                    className="border rounded px-3 py-1 text-sm"
                  >
                    <option value="all">All Types</option>
                    <option value="document">Documents</option>
                    <option value="code">Code</option>
                    <option value="web">Web Sources</option>
                    <option value="note">Notes</option>
                  </select>
                </div>
                <div className="text-sm text-gray-500">
                  {filteredItems.length} items
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Knowledge Items */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {filteredItems.map((item) => (
              <Card key={item.id} className="hover:shadow-md transition-shadow">
                <CardHeader className="pb-3">
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-2">
                      {getTypeIcon(item.type)}
                      <CardTitle className="text-lg leading-tight">{item.title}</CardTitle>
                    </div>
                    <Button variant="ghost" size="sm">
                      <MoreHorizontal className="w-4 h-4" />
                    </Button>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge className={getTypeColor(item.type)}>
                      {item.type}
                    </Badge>
                    <div className="flex items-center gap-1">
                      {getStatusIcon(item.embedding_status)}
                      <span className="text-xs text-gray-500">{item.embedding_status}</span>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3 line-clamp-2">
                    {item.content_preview}
                  </p>
                  
                  <div className="flex flex-wrap gap-1 mb-3">
                    {item.tags.slice(0, 3).map((tag, index) => (
                      <Badge key={index} variant="outline" className="text-xs">
                        <Tag className="w-2 h-2 mr-1" />
                        {tag}
                      </Badge>
                    ))}
                    {item.tags.length > 3 && (
                      <Badge variant="outline" className="text-xs">
                        +{item.tags.length - 3}
                      </Badge>
                    )}
                  </div>

                  <div className="flex items-center justify-between text-xs text-gray-500 mb-3">
                    <span>Used {item.usage_count} times</span>
                    <span>Updated {formatDate(item.updated_at)}</span>
                  </div>

                  <div className="flex gap-2">
                    <Button variant="outline" size="sm" className="flex-1">
                      <Eye className="w-3 h-3 mr-1" />
                      View
                    </Button>
                    <Button variant="outline" size="sm">
                      <Edit className="w-3 h-3" />
                    </Button>
                    <Button variant="outline" size="sm">
                      <Share className="w-3 h-3" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {filteredItems.length === 0 && (
            <Card className="p-8">
              <div className="text-center text-gray-500">
                <BookOpen className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <h3 className="text-lg font-medium mb-2">No Knowledge Items</h3>
                <p>Upload documents or add knowledge to get started.</p>
              </div>
            </Card>
          )}
        </TabsContent>

        {/* Search Tab */}
        <TabsContent value="search" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Search className="w-5 h-5 text-blue-600" />
                Semantic Search
              </CardTitle>
              <CardDescription>
                Search through your knowledge base using AI-powered semantic understanding
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex gap-2">
                <Input
                  placeholder="Search for information, code, or concepts..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && performSearch()}
                  className="flex-1"
                />
                <Button onClick={performSearch} disabled={isSearching || !searchQuery.trim()}>
                  {isSearching ? 'Searching...' : 'Search'}
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Search Results */}
          {searchResults.length > 0 && (
            <div className="space-y-3">
              <h3 className="text-lg font-semibold">Search Results ({searchResults.length})</h3>
              {searchResults.map((result) => (
                <Card key={result.id} className="hover:shadow-md transition-shadow">
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        {getTypeIcon(result.type)}
                        <h4 className="font-medium">{result.title}</h4>
                      </div>
                      <Badge variant="outline">
                        {Math.round(result.relevance_score * 100)}% match
                      </Badge>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                      {result.content}
                    </p>
                    <div className="flex items-center justify-between text-xs text-gray-500">
                      <span>Source: {result.source}</span>
                      <Button variant="outline" size="sm">
                        <Eye className="w-3 h-3 mr-1" />
                        View Full
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}

          {searchQuery && searchResults.length === 0 && !isSearching && (
            <Card className="p-8">
              <div className="text-center text-gray-500">
                <Search className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <h3 className="text-lg font-medium mb-2">No Results Found</h3>
                <p>Try adjusting your search terms or browse the knowledge base.</p>
              </div>
            </Card>
          )}
        </TabsContent>

        {/* Analytics Tab */}
        <TabsContent value="analytics" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Usage Statistics */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="w-5 h-5 text-green-600" />
                  Usage Statistics
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span>Documents</span>
                    <span className="font-medium">{stats.documents}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Code Snippets</span>
                    <span className="font-medium">{stats.code_snippets}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Web Sources</span>
                    <span className="font-medium">{stats.web_sources}</span>
                  </div>
                  <div className="pt-4 border-t">
                    <div className="flex justify-between items-center">
                      <span>Storage Efficiency</span>
                      <span className="font-medium text-green-600">Good</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Popular Content */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="w-5 h-5 text-purple-600" />
                  Most Accessed
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {knowledgeItems
                    .sort((a, b) => b.usage_count - a.usage_count)
                    .slice(0, 5)
                    .map((item, index) => (
                      <div key={item.id} className="flex items-center gap-3">
                        <div className="w-6 h-6 rounded-full bg-purple-100 dark:bg-purple-900/20 flex items-center justify-center text-xs font-bold text-purple-600">
                          {index + 1}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="font-medium text-sm truncate">{item.title}</div>
                          <div className="text-xs text-gray-500">{item.usage_count} uses</div>
                        </div>
                      </div>
                    ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default KnowledgeManagement;