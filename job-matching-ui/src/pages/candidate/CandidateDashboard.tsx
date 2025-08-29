import { useEffect, useState } from "react";
import { api } from "../../api/mockApi";
import { useAuth } from "../../context/AuthContext";
import type { CV } from "../../types";
import { Link } from "react-router-dom";

export default function CandidateDashboard() {
  const { user } = useAuth();
  const [cvs, setCVs] = useState<CV[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!user) return;
    api.listCVsByUser(user.id).then(setCVs).finally(()=>setLoading(false));
  }, [user]);

  if (!user) return null;

  return (
    <div className="container">
      <div className="card">
        <h2>Ứng viên • CV của tôi</h2>
        <div className="mt-3" style={{display:'flex', gap:12, flexWrap:'wrap'}}>
          <Link to="/candidate/upload" className="btn success">＋ Upload CV (PDF/DOCX)</Link>
          <Link to="/candidate/edit" className="btn">＋ Tạo/Chỉnh CV (JSON)</Link>
        </div>

        <hr/>
        {loading ? <p>Đang tải…</p> : (
          cvs.length === 0 ? <p>Chưa có CV.</p> : (
            <div className="row mt-3">
              {cvs.map(cv => (
                <div key={cv.id} className="card">
                  <h3 className="mb-0">{cv.title}</h3>
                  <p className="mt-2" style={{color:'var(--muted)'}}>{cv.summary}</p>
                  <div className="mt-2">
                    <span className="badge">Skills: {cv.skills.join(', ') || '—'}</span>
                  </div>
                  <div className="mt-3" style={{display:'flex', gap:8, flexWrap:'wrap'}}>
                    <Link to={`/candidate/cv/${cv.id}/edit`} className="btn">Chỉnh sửa</Link>
                    {cv.fileBase64 && <a className="btn" href={cv.fileBase64} download={`${cv.title}.txt`}>Tải file</a>}
                    <span className="badge">Cập nhật: {new Date(cv.updatedAt).toLocaleString()}</span>
                  </div>
                </div>
              ))}
            </div>
          )
        )}
      </div>
    </div>
  );
}
