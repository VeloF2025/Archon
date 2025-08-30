import React, { useState } from 'react';

interface LoginProps {
  onLogin?: (token: string) => void;
}

const Login: React.FC<LoginProps> = ({ onLogin }) => {
  const [credentials, setCredentials] = useState({ username: '', password: '' });
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    // TODO: Implement login API call
    console.log('Login attempt:', credentials);
  };

  return (
    <form onSubmit={handleSubmit}>
      <div>
        <label>Username:</label>
        <input 
          type="text" 
          value={credentials.username}
          onChange={(e) => setCredentials({...credentials, username: e.target.value})}
        />
      </div>
      <div>
        <label>Password:</label>
        <input 
          type="password"
          value={credentials.password} 
          onChange={(e) => setCredentials({...credentials, password: e.target.value})}
        />
      </div>
      <button type="submit">Login</button>
    </form>
  );
};

export default Login;
