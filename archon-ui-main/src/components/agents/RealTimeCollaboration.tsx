import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Input } from '../ui/input';
import { Progress } from '../ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { useSocket } from '../../hooks/useSocket';
import { useToast } from '../../hooks/useToast';
import { 
  Users,
  MessageSquare,
  Share2,
  Globe,
  Clock,
  Zap,
  CheckCircle,
  AlertCircle,
  Eye,
  Send,
  Activity,
  Mic,
  MicOff,
  Video,
  VideoOff,
  Settings
} from 'lucide-react';

interface CollaborationSession {
  id: string;
  name: string;
  participants: string[];
  active_agents: string[];
  created_at: string;
  status: 'active' | 'paused' | 'completed';
  task_description: string;
  progress: number;
}

interface AgentActivity {
  agent_id: string;
  agent_name: string;
  action: string;
  timestamp: string;
  details: string;
  status: 'in_progress' | 'completed' | 'failed';
}

interface RealTimeCollaborationProps {
  agents: any[];
}

export const RealTimeCollaboration: React.FC<RealTimeCollaborationProps> = ({ agents }) => {
  const [sessions, setSessions] = useState<CollaborationSession[]>([]);
  const [activities, setActivities] = useState<AgentActivity[]>([]);
  const [selectedSession, setSelectedSession] = useState<CollaborationSession | null>(null);
  const [newSessionName, setNewSessionName] = useState('');
  const [isCreatingSession, setIsCreatingSession] = useState(false);
  const [message, setMessage] = useState('');
  const [isObserving, setIsObserving] = useState(false);
  const [connectedUsers, setConnectedUsers] = useState<string[]>([]);

  const socket = useSocket();
  const { toast } = useToast();

  useEffect(() => {
    if (!socket) return;

    // Socket event listeners for real-time collaboration
    const handleSessionUpdate = (session: CollaborationSession) => {
      setSessions(prev => prev.map(s => s.id === session.id ? session : s));
    };

    const handleNewActivity = (activity: AgentActivity) => {
      setActivities(prev => [activity, ...prev.slice(0, 49)]);
    };

    const handleUserJoined = (data: { sessionId: string; userId: string }) => {
      setConnectedUsers(prev => [...prev, data.userId]);
      toast({
        title: "User Joined",
        description: `${data.userId} joined the collaboration`,
        variant: "default"
      });
    };

    const handleUserLeft = (data: { sessionId: string; userId: string }) => {
      setConnectedUsers(prev => prev.filter(u => u !== data.userId));
    };

    // Create stable handler references for socket events
    const sessionUpdateHandler = (message: any) => handleSessionUpdate(message.data);
    const activityHandler = (message: any) => handleNewActivity(message.data);
    const userJoinedHandler = (message: any) => handleUserJoined(message.data);
    const userLeftHandler = (message: any) => handleUserLeft(message.data);

    socket.addMessageHandler('collaboration_session_update', sessionUpdateHandler);
    socket.addMessageHandler('agent_activity', activityHandler);
    socket.addMessageHandler('user_joined_session', userJoinedHandler);
    socket.addMessageHandler('user_left_session', userLeftHandler);

    return () => {
      socket.removeMessageHandler('collaboration_session_update', sessionUpdateHandler);
      socket.removeMessageHandler('agent_activity', activityHandler);
      socket.removeMessageHandler('user_joined_session', userJoinedHandler);
      socket.removeMessageHandler('user_left_session', userLeftHandler);
    };
  }, [socket, toast]);

  const createSession = async () => {
    if (!newSessionName.trim()) return;
    
    setIsCreatingSession(true);
    try {
      // Simulate session creation
      const newSession: CollaborationSession = {
        id: `session_${Date.now()}`,
        name: newSessionName,
        participants: ['current_user'],
        active_agents: [],
        created_at: new Date().toISOString(),
        status: 'active',
        task_description: '',
        progress: 0
      };
      
      setSessions(prev => [newSession, ...prev]);
      setNewSessionName('');
      
      toast({
        title: "Session Created",
        description: `Collaboration session "${newSessionName}" created successfully`,
        variant: "success"
      });
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to create collaboration session",
        variant: "destructive"
      });
    } finally {
      setIsCreatingSession(false);
    }
  };

  const joinSession = (session: CollaborationSession) => {
    setSelectedSession(session);
    setIsObserving(true);
    
    if (socket) {
      socket.emit('join_collaboration_session', { sessionId: session.id });
    }
    
    toast({
      title: "Joined Session",
      description: `You've joined "${session.name}"`,
      variant: "success"
    });
  };

  const leaveSession = () => {
    if (selectedSession && socket) {
      socket.emit('leave_collaboration_session', { sessionId: selectedSession.id });
    }
    
    setSelectedSession(null);
    setIsObserving(false);
  };

  const sendMessage = () => {
    if (!message.trim() || !selectedSession) return;
    
    if (socket) {
      socket.emit('collaboration_message', {
        sessionId: selectedSession.id,
        message: message.trim(),
        timestamp: new Date().toISOString()
      });
    }
    
    setMessage('');
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'paused':
        return <Clock className="w-4 h-4 text-yellow-500" />;
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-blue-500" />;
      default:
        return <AlertCircle className="w-4 h-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400';
      case 'paused':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400';
      case 'completed':
        return 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400';
    }
  };

  const formatTimeAgo = (timestamp: string) => {
    const diff = Date.now() - new Date(timestamp).getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (days > 0) return `${days}d ago`;
    if (hours > 0) return `${hours}h ago`;
    if (minutes > 0) return `${minutes}m ago`;
    return 'Just now';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Users className="w-8 h-8 text-blue-600" />
            Real-Time Collaboration
          </h2>
          <p className="text-gray-600 dark:text-gray-400">
            Collaborate with agents and team members in real-time
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="flex items-center gap-1">
            <Globe className="w-3 h-3" />
            {connectedUsers.length} online
          </Badge>
          <Button variant="outline" className="flex items-center gap-2">
            <Settings className="w-4 h-4" />
            Settings
          </Button>
        </div>
      </div>

      <Tabs defaultValue="sessions" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="sessions">🤝 Sessions</TabsTrigger>
          <TabsTrigger value="activity">⚡ Live Activity</TabsTrigger>
          <TabsTrigger value="workspace">🖥️ Workspace</TabsTrigger>
        </TabsList>

        {/* Sessions Tab */}
        <TabsContent value="sessions" className="space-y-4">
          {/* Create New Session */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Share2 className="w-5 h-5 text-blue-600" />
                Create Collaboration Session
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex gap-2">
                <Input
                  placeholder="Enter session name..."
                  value={newSessionName}
                  onChange={(e) => setNewSessionName(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && createSession()}
                  className="flex-1"
                />
                <Button 
                  onClick={createSession}
                  disabled={!newSessionName.trim() || isCreatingSession}
                >
                  {isCreatingSession ? 'Creating...' : 'Create'}
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Active Sessions */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {sessions.map((session) => (
              <Card key={session.id} className="hover:shadow-md transition-shadow">
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg">{session.name}</CardTitle>
                    <Badge className={getStatusColor(session.status)}>
                      {getStatusIcon(session.status)}
                      {session.status}
                    </Badge>
                  </div>
                  <CardDescription>
                    Created {formatTimeAgo(session.created_at)}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Progress</span>
                        <span>{session.progress}%</span>
                      </div>
                      <Progress value={session.progress} className="h-2" />
                    </div>
                    
                    <div className="flex items-center justify-between text-sm">
                      <span className="flex items-center gap-1">
                        <Users className="w-3 h-3" />
                        {session.participants.length} participants
                      </span>
                      <span className="flex items-center gap-1">
                        <Zap className="w-3 h-3" />
                        {session.active_agents.length} agents
                      </span>
                    </div>

                    <Button 
                      className="w-full" 
                      variant={selectedSession?.id === session.id ? "default" : "outline"}
                      onClick={() => selectedSession?.id === session.id ? leaveSession() : joinSession(session)}
                    >
                      {selectedSession?.id === session.id ? (
                        <>
                          <Eye className="w-4 h-4 mr-2" />
                          Leave Session
                        </>
                      ) : (
                        <>
                          <Share2 className="w-4 h-4 mr-2" />
                          Join Session
                        </>
                      )}
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {sessions.length === 0 && (
            <Card className="p-8">
              <div className="text-center text-gray-500">
                <Users className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <h3 className="text-lg font-medium mb-2">No Active Sessions</h3>
                <p>Create a new collaboration session to get started.</p>
              </div>
            </Card>
          )}
        </TabsContent>

        {/* Live Activity Tab */}
        <TabsContent value="activity" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Zap className="w-5 h-5 text-yellow-600" />
                Live Agent Activity
              </CardTitle>
              <CardDescription>
                Real-time updates from active agents
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {activities.map((activity, index) => (
                  <div key={index} className="flex items-start gap-3 p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-100 dark:bg-blue-900/20 flex items-center justify-center">
                      {activity.status === 'completed' && <CheckCircle className="w-4 h-4 text-green-500" />}
                      {activity.status === 'in_progress' && <Clock className="w-4 h-4 text-yellow-500" />}
                      {activity.status === 'failed' && <AlertCircle className="w-4 h-4 text-red-500" />}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-medium text-sm">{activity.agent_name}</span>
                        <Badge variant="outline" size="sm">{activity.status}</Badge>
                        <span className="text-xs text-gray-500">{formatTimeAgo(activity.timestamp)}</span>
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">{activity.action}</p>
                      {activity.details && (
                        <p className="text-xs text-gray-500 mt-1">{activity.details}</p>
                      )}
                    </div>
                  </div>
                ))}
                
                {activities.length === 0 && (
                  <div className="text-center py-8 text-gray-500">
                    <Activity className="w-8 h-8 mx-auto mb-2 opacity-50" />
                    <p>No recent activity</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Workspace Tab */}
        <TabsContent value="workspace" className="space-y-4">
          {selectedSession ? (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Session Info */}
              <Card className="lg:col-span-2">
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <span>{selectedSession.name}</span>
                    <Badge className={getStatusColor(selectedSession.status)}>
                      {selectedSession.status}
                    </Badge>
                  </CardTitle>
                  <CardDescription>
                    Collaborative workspace for real-time development
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex items-center gap-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                      <div className="flex gap-2">
                        <Button variant="outline" size="sm">
                          <Mic className="w-4 h-4" />
                        </Button>
                        <Button variant="outline" size="sm">
                          <Video className="w-4 h-4" />
                        </Button>
                        <Button variant="outline" size="sm">
                          <Share2 className="w-4 h-4" />
                        </Button>
                      </div>
                      <div className="flex-1">
                        <Progress value={selectedSession.progress} className="h-2" />
                      </div>
                      <span className="text-sm font-medium">{selectedSession.progress}%</span>
                    </div>

                    {/* Message Input */}
                    <div className="flex gap-2">
                      <Input
                        placeholder="Type a message..."
                        value={message}
                        onChange={(e) => setMessage(e.target.value)}
                        onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                        className="flex-1"
                      />
                      <Button onClick={sendMessage} disabled={!message.trim()}>
                        <Send className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Participants */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Users className="w-5 h-5" />
                    Participants
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {selectedSession.participants.map((participant, index) => (
                      <div key={index} className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-full bg-blue-100 dark:bg-blue-900/20 flex items-center justify-center">
                          <span className="text-sm font-medium">
                            {participant.charAt(0).toUpperCase()}
                          </span>
                        </div>
                        <span className="text-sm">{participant}</span>
                        <div className="w-2 h-2 bg-green-500 rounded-full ml-auto"></div>
                      </div>
                    ))}
                    
                    <div className="border-t pt-3 mt-3">
                      <h4 className="text-sm font-medium mb-2">Active Agents</h4>
                      {selectedSession.active_agents.length > 0 ? (
                        selectedSession.active_agents.map((agentId, index) => (
                          <div key={index} className="flex items-center gap-3 mb-2">
                            <div className="w-6 h-6 rounded bg-purple-100 dark:bg-purple-900/20 flex items-center justify-center">
                              <Zap className="w-3 h-3 text-purple-600" />
                            </div>
                            <span className="text-sm">{agentId}</span>
                          </div>
                        ))
                      ) : (
                        <p className="text-sm text-gray-500">No agents active</p>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          ) : (
            <Card className="p-8">
              <div className="text-center text-gray-500">
                <MessageSquare className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <h3 className="text-lg font-medium mb-2">No Session Selected</h3>
                <p>Join a collaboration session to access the workspace.</p>
              </div>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default RealTimeCollaboration;