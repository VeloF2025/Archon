/**
 * Access Control Manager Component
 *
 * Comprehensive access control management interface
 * Provides role-based access control (RBAC) and attribute-based access control (ABAC) management
 */

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Users, Shield, Plus, Edit, Trash2, Key, Search, Filter } from 'lucide-react';

interface Subject {
  id: string;
  subject_type: string;
  name: string;
  roles: string[];
  attributes: Record<string, any>;
  created_at: string;
  last_active: string;
}

interface Role {
  id: string;
  name: string;
  description: string;
  permissions: string[];
  parent_roles: string[];
  created_at: string;
}

interface Permission {
  id: string;
  name: string;
  resource: string;
  action: string;
  description: string;
  conditions: string[];
}

interface AccessLog {
  id: string;
  subject_id: string;
  resource: string;
  action: string;
  decision: 'permit' | 'deny';
  timestamp: string;
  reason: string;
  context: Record<string, any>;
}

export const AccessControlManager: React.FC = () => {
  const [subjects, setSubjects] = useState<Subject[]>([]);
  const [roles, setRoles] = useState<Role[]>([]);
  const [permissions, setPermissions] = useState<Permission[]>([]);
  const [accessLogs, setAccessLogs] = useState<AccessLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedTab, setSelectedTab] = useState('subjects');

  // Dialog states
  const [showCreateSubject, setShowCreateSubject] = useState(false);
  const [showCreateRole, setShowCreateRole] = useState(false);
  const [showCreatePermission, setShowCreatePermission] = useState(false);

  // Form states
  const [newSubject, setNewSubject] = useState({
    id: '',
    subject_type: 'user',
    name: '',
    roles: [] as string[],
    attributes: {}
  });

  const [newRole, setNewRole] = useState({
    id: '',
    name: '',
    description: '',
    permissions: [] as string[],
    parent_roles: [] as string[]
  });

  const [newPermission, setNewPermission] = useState({
    id: '',
    name: '',
    resource: '',
    action: '',
    description: '',
    conditions: [] as string[]
  });

  useEffect(() => {
    loadAccessControlData();
  }, []);

  const loadAccessControlData = async () => {
    try {
      const [subjectsRes, rolesRes, permissionsRes, logsRes] = await Promise.all([
        fetch('/api/security/access/subjects'),
        fetch('/api/security/access/roles'),
        fetch('/api/security/access/permissions'),
        fetch('/api/security/access/logs?limit=100')
      ]);

      if (subjectsRes.ok) {
        const subjectsData = await subjectsRes.json();
        setSubjects(subjectsData.subjects || []);
      }

      if (rolesRes.ok) {
        const rolesData = await rolesRes.json();
        setRoles(rolesData.roles || []);
      }

      if (permissionsRes.ok) {
        const permissionsData = await permissionsRes.json();
        setPermissions(permissionsData.permissions || []);
      }

      if (logsRes.ok) {
        const logsData = await logsRes.json();
        setAccessLogs(logsData.logs || []);
      }
    } catch (error) {
      console.error('Error loading access control data:', error);
    } finally {
      setLoading(false);
    }
  };

  const createSubject = async () => {
    try {
      const response = await fetch('/api/security/access/subjects', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newSubject)
      });

      if (response.ok) {
        await loadAccessControlData();
        setShowCreateSubject(false);
        setNewSubject({ id: '', subject_type: 'user', name: '', roles: [], attributes: {} });
      }
    } catch (error) {
      console.error('Error creating subject:', error);
    }
  };

  const createRole = async () => {
    try {
      const response = await fetch('/api/security/access/roles', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newRole)
      });

      if (response.ok) {
        await loadAccessControlData();
        setShowCreateRole(false);
        setNewRole({ id: '', name: '', description: '', permissions: [], parent_roles: [] });
      }
    } catch (error) {
      console.error('Error creating role:', error);
    }
  };

  const assignRoleToSubject = async (subjectId: string, roleId: string) => {
    try {
      const response = await fetch(`/api/security/access/subjects/${subjectId}/roles/${roleId}`, {
        method: 'POST'
      });

      if (response.ok) {
        await loadAccessControlData();
      }
    } catch (error) {
      console.error('Error assigning role:', error);
    }
  };

  const revokeRoleFromSubject = async (subjectId: string, roleId: string) => {
    try {
      const response = await fetch(`/api/security/access/subjects/${subjectId}/roles/${roleId}`, {
        method: 'DELETE'
      });

      if (response.ok) {
        await loadAccessControlData();
      }
    } catch (error) {
      console.error('Error revoking role:', error);
    }
  };

  const checkAccess = async (subjectId: string, resource: string, action: string) => {
    try {
      const response = await fetch('/api/security/access/check', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          subject_id: subjectId,
          resource,
          action,
          context: {}
        })
      });

      if (response.ok) {
        const result = await response.json();
        return result.decision === 'permit';
      }
      return false;
    } catch (error) {
      console.error('Error checking access:', error);
      return false;
    }
  };

  const filteredSubjects = subjects.filter(subject =>
    subject.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    subject.id.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const filteredRoles = roles.filter(role =>
    role.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    role.description.toLowerCase().includes(searchTerm.toLowerCase())
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Access Control Manager</h1>
          <p className="text-gray-600">Manage subjects, roles, permissions, and access policies</p>
        </div>
        <div className="flex items-center space-x-2">
          <div className="relative">
            <Search className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
            <Input
              placeholder="Search..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10 w-64"
            />
          </div>
          <Button onClick={loadAccessControlData} variant="outline">
            Refresh
          </Button>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Subjects</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{subjects.length}</div>
            <p className="text-xs text-muted-foreground">
              Users and systems
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Roles</CardTitle>
            <Shield className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{roles.length}</div>
            <p className="text-xs text-muted-foreground">
              Access roles defined
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Permissions</CardTitle>
            <Key className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{permissions.length}</div>
            <p className="text-xs text-muted-foreground">
              Granular permissions
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Access Events (24h)</CardTitle>
            <Filter className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{accessLogs.length}</div>
            <p className="text-xs text-muted-foreground">
              Recent access decisions
            </p>
          </CardContent>
        </Card>
      </div>

      <Tabs value={selectedTab} onValueChange={setSelectedTab} className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="subjects">Subjects</TabsTrigger>
          <TabsTrigger value="roles">Roles</TabsTrigger>
          <TabsTrigger value="permissions">Permissions</TabsTrigger>
          <TabsTrigger value="logs">Access Logs</TabsTrigger>
        </TabsList>

        <TabsContent value="subjects">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                Subjects
                <Dialog open={showCreateSubject} onOpenChange={setShowCreateSubject}>
                  <DialogTrigger asChild>
                    <Button>
                      <Plus className="h-4 w-4 mr-2" />
                      Create Subject
                    </Button>
                  </DialogTrigger>
                  <DialogContent>
                    <DialogHeader>
                      <DialogTitle>Create New Subject</DialogTitle>
                      <DialogDescription>Add a new user or system subject</DialogDescription>
                    </DialogHeader>
                    <div className="space-y-4">
                      <div>
                        <Label htmlFor="subject-id">Subject ID</Label>
                        <Input
                          id="subject-id"
                          value={newSubject.id}
                          onChange={(e) => setNewSubject({...newSubject, id: e.target.value})}
                          placeholder="Enter unique subject identifier"
                        />
                      </div>
                      <div>
                        <Label htmlFor="subject-type">Subject Type</Label>
                        <Select onValueChange={(value) => setNewSubject({...newSubject, subject_type: value})}>
                          <SelectTrigger>
                            <SelectValue placeholder="Select subject type" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="user">User</SelectItem>
                            <SelectItem value="system">System</SelectItem>
                            <SelectItem value="service">Service</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div>
                        <Label htmlFor="subject-name">Name</Label>
                        <Input
                          id="subject-name"
                          value={newSubject.name}
                          onChange={(e) => setNewSubject({...newSubject, name: e.target.value})}
                          placeholder="Enter subject name"
                        />
                      </div>
                      <div className="flex justify-end space-x-2">
                        <Button variant="outline" onClick={() => setShowCreateSubject(false)}>
                          Cancel
                        </Button>
                        <Button onClick={createSubject}>
                          Create Subject
                        </Button>
                      </div>
                    </div>
                  </DialogContent>
                </Dialog>
              </CardTitle>
              <CardDescription>Manage users and system subjects</CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>ID</TableHead>
                    <TableHead>Name</TableHead>
                    <TableHead>Type</TableHead>
                    <TableHead>Roles</TableHead>
                    <TableHead>Last Active</TableHead>
                    <TableHead>Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredSubjects.map((subject) => (
                    <TableRow key={subject.id}>
                      <TableCell className="font-medium">{subject.id}</TableCell>
                      <TableCell>{subject.name}</TableCell>
                      <TableCell>
                        <Badge variant="outline">{subject.subject_type}</Badge>
                      </TableCell>
                      <TableCell>
                        <div className="flex flex-wrap gap-1">
                          {subject.roles.map((role) => (
                            <Badge key={role} variant="secondary" className="text-xs">
                              {role}
                            </Badge>
                          ))}
                        </div>
                      </TableCell>
                      <TableCell>
                        {new Date(subject.last_active).toLocaleDateString()}
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center space-x-2">
                          <Button size="sm" variant="outline">
                            <Edit className="h-3 w-3" />
                          </Button>
                          <Button size="sm" variant="outline">
                            <Trash2 className="h-3 w-3" />
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="roles">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                Roles
                <Dialog open={showCreateRole} onOpenChange={setShowCreateRole}>
                  <DialogTrigger asChild>
                    <Button>
                      <Plus className="h-4 w-4 mr-2" />
                      Create Role
                    </Button>
                  </DialogTrigger>
                  <DialogContent>
                    <DialogHeader>
                      <DialogTitle>Create New Role</DialogTitle>
                      <DialogDescription>Define a new access role</DialogDescription>
                    </DialogHeader>
                    <div className="space-y-4">
                      <div>
                        <Label htmlFor="role-id">Role ID</Label>
                        <Input
                          id="role-id"
                          value={newRole.id}
                          onChange={(e) => setNewRole({...newRole, id: e.target.value})}
                          placeholder="Enter unique role identifier"
                        />
                      </div>
                      <div>
                        <Label htmlFor="role-name">Role Name</Label>
                        <Input
                          id="role-name"
                          value={newRole.name}
                          onChange={(e) => setNewRole({...newRole, name: e.target.value})}
                          placeholder="Enter role name"
                        />
                      </div>
                      <div>
                        <Label htmlFor="role-description">Description</Label>
                        <Textarea
                          id="role-description"
                          value={newRole.description}
                          onChange={(e) => setNewRole({...newRole, description: e.target.value})}
                          placeholder="Describe the role's purpose"
                        />
                      </div>
                      <div className="flex justify-end space-x-2">
                        <Button variant="outline" onClick={() => setShowCreateRole(false)}>
                          Cancel
                        </Button>
                        <Button onClick={createRole}>
                          Create Role
                        </Button>
                      </div>
                    </div>
                  </DialogContent>
                </Dialog>
              </CardTitle>
              <CardDescription>Define roles and permission sets</CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>ID</TableHead>
                    <TableHead>Name</TableHead>
                    <TableHead>Description</TableHead>
                    <TableHead>Permissions</TableHead>
                    <TableHead>Parent Roles</TableHead>
                    <TableHead>Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredRoles.map((role) => (
                    <TableRow key={role.id}>
                      <TableCell className="font-medium">{role.id}</TableCell>
                      <TableCell>{role.name}</TableCell>
                      <TableCell className="max-w-xs truncate">{role.description}</TableCell>
                      <TableCell>
                        <div className="flex flex-wrap gap-1">
                          {role.permissions.slice(0, 3).map((permission) => (
                            <Badge key={permission} variant="secondary" className="text-xs">
                              {permission}
                            </Badge>
                          ))}
                          {role.permissions.length > 3 && (
                            <Badge variant="outline" className="text-xs">
                              +{role.permissions.length - 3} more
                            </Badge>
                          )}
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="flex flex-wrap gap-1">
                          {role.parent_roles.map((parent) => (
                            <Badge key={parent} variant="outline" className="text-xs">
                              {parent}
                            </Badge>
                          ))}
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center space-x-2">
                          <Button size="sm" variant="outline">
                            <Edit className="h-3 w-3" />
                          </Button>
                          <Button size="sm" variant="outline">
                            <Trash2 className="h-3 w-3" />
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="permissions">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                Permissions
                <Dialog open={showCreatePermission} onOpenChange={setShowCreatePermission}>
                  <DialogTrigger asChild>
                    <Button>
                      <Plus className="h-4 w-4 mr-2" />
                      Create Permission
                    </Button>
                  </DialogTrigger>
                  <DialogContent>
                    <DialogHeader>
                      <DialogTitle>Create New Permission</DialogTitle>
                      <DialogDescription>Define a granular permission</DialogDescription>
                    </DialogHeader>
                    <div className="space-y-4">
                      <div>
                        <Label htmlFor="permission-id">Permission ID</Label>
                        <Input
                          id="permission-id"
                          value={newPermission.id}
                          onChange={(e) => setNewPermission({...newPermission, id: e.target.value})}
                          placeholder="Enter unique permission identifier"
                        />
                      </div>
                      <div>
                        <Label htmlFor="permission-name">Permission Name</Label>
                        <Input
                          id="permission-name"
                          value={newPermission.name}
                          onChange={(e) => setNewPermission({...newPermission, name: e.target.value})}
                          placeholder="Enter permission name"
                        />
                      </div>
                      <div>
                        <Label htmlFor="permission-resource">Resource</Label>
                        <Input
                          id="permission-resource"
                          value={newPermission.resource}
                          onChange={(e) => setNewPermission({...newPermission, resource: e.target.value})}
                          placeholder="e.g., /api/users"
                        />
                      </div>
                      <div>
                        <Label htmlFor="permission-action">Action</Label>
                        <Select onValueChange={(value) => setNewPermission({...newPermission, action: value})}>
                          <SelectTrigger>
                            <SelectValue placeholder="Select action" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="read">Read</SelectItem>
                            <SelectItem value="write">Write</SelectItem>
                            <SelectItem value="delete">Delete</SelectItem>
                            <SelectItem value="execute">Execute</SelectItem>
                            <SelectItem value="admin">Admin</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="flex justify-end space-x-2">
                        <Button variant="outline" onClick={() => setShowCreatePermission(false)}>
                          Cancel
                        </Button>
                        <Button onClick={() => {/* TODO: Implement create permission */}}>
                          Create Permission
                        </Button>
                      </div>
                    </div>
                  </DialogContent>
                </Dialog>
              </CardTitle>
              <CardDescription>Granular permission definitions</CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>ID</TableHead>
                    <TableHead>Name</TableHead>
                    <TableHead>Resource</TableHead>
                    <TableHead>Action</TableHead>
                    <TableHead>Description</TableHead>
                    <TableHead>Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {permissions.map((permission) => (
                    <TableRow key={permission.id}>
                      <TableCell className="font-medium">{permission.id}</TableCell>
                      <TableCell>{permission.name}</TableCell>
                      <TableCell>
                        <Badge variant="outline">{permission.resource}</Badge>
                      </TableCell>
                      <TableCell>
                        <Badge variant="secondary">{permission.action}</Badge>
                      </TableCell>
                      <TableCell className="max-w-xs truncate">{permission.description}</TableCell>
                      <TableCell>
                        <div className="flex items-center space-x-2">
                          <Button size="sm" variant="outline">
                            <Edit className="h-3 w-3" />
                          </Button>
                          <Button size="sm" variant="outline">
                            <Trash2 className="h-3 w-3" />
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="logs">
          <Card>
            <CardHeader>
              <CardTitle>Access Logs</CardTitle>
              <CardDescription>Recent access control decisions and events</CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Timestamp</TableHead>
                    <TableHead>Subject</TableHead>
                    <TableHead>Resource</TableHead>
                    <TableHead>Action</TableHead>
                    <TableHead>Decision</TableHead>
                    <TableHead>Reason</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {accessLogs.slice(0, 50).map((log) => (
                    <TableRow key={log.id}>
                      <TableCell className="font-medium">
                        {new Date(log.timestamp).toLocaleString()}
                      </TableCell>
                      <TableCell>{log.subject_id}</TableCell>
                      <TableCell>{log.resource}</TableCell>
                      <TableCell>{log.action}</TableCell>
                      <TableCell>
                        <Badge variant={log.decision === 'permit' ? 'default' : 'destructive'}>
                          {log.decision}
                        </Badge>
                      </TableCell>
                      <TableCell className="max-w-xs truncate">{log.reason}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};