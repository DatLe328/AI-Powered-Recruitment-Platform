import { createBrowserRouter } from "react-router-dom";
import Home from "./pages/Home";
import Login from "./pages/Login";
import Register from "./pages/Register";
import NotFound from "./pages/NotFound";
import CandidateDashboard from "./pages/candidate/CandidateDashboard";
import UploadCV from "./pages/candidate/UploadCV";
import EditCV from "./pages/candidate/EditCV";
import EmployerDashboard from "./pages/employer/EmployerDashboard";
import PostJob from "./pages/employer/PostJob";
import ManageJobs from "./pages/employer/ManageJobs";
import { ProtectedRoute } from "./components/ProtectedRoute";

export const router = createBrowserRouter([
  { path: "/", element: <Home/> },
  { path: "/login", element: <Login/> },
  { path: "/register", element: <Register/> },

  { path: "/candidate", element: <ProtectedRoute role="candidate"><CandidateDashboard/></ProtectedRoute> },
  { path: "/candidate/upload", element: <ProtectedRoute role="candidate"><UploadCV/></ProtectedRoute> },
  { path: "/candidate/edit", element: <ProtectedRoute role="candidate"><EditCV/></ProtectedRoute> },
  { path: "/candidate/cv/:id/edit", element: <ProtectedRoute role="candidate"><EditCV/></ProtectedRoute> },

  { path: "/employer", element: <ProtectedRoute role="employer"><EmployerDashboard/></ProtectedRoute> },
  { path: "/employer/post", element: <ProtectedRoute role="employer"><PostJob/></ProtectedRoute> },
  { path: "/employer/jobs", element: <ProtectedRoute role="employer"><ManageJobs/></ProtectedRoute> },

  { path: "*", element: <NotFound/> }
]);
