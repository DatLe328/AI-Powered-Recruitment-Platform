import React from "react";
import { Navigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import type { Role } from "../types";

export const ProtectedRoute: React.FC<{ children: React.ReactNode; role?: Role }> = ({ children, role }) => {
  const { user, loading } = useAuth();

  if (loading) return <div className="container"><div className="card">Đang tải…</div></div>;
  if (!user) return <Navigate to="/login" replace />;

  if (role && user.role !== role) {
    // Không đúng vai trò → chuyển về dashboard phù hợp
    return <Navigate to={user.role === 'candidate' ? "/candidate" : "/employer"} replace />;
  }
  return <>{children}</>;
};
