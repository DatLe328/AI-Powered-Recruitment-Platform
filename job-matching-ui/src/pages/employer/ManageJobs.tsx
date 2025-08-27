import { useEffect, useState } from "react";
import { api } from "../../api/mockApi";
import { useAuth } from "../../context/AuthContext";
import type { Job } from "../../types";

export default function ManageJobs() {
  const { user } = useAuth();
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!user) return;
    api.listJobsByEmployer(user.id).then(setJobs).finally(()=>setLoading(false));
  }, [user]);

  if (!user) return null;

  return (
    <div className="container">
      <div className="card">
        <h2>Danh sách tin tuyển dụng</h2>
        {loading ? <p>Đang tải…</p> : jobs.length === 0 ? <p>Chưa có tin nào.</p> : (
          <div className="row mt-3">
            {jobs.map(j => (
              <div key={j.id} className="card">
                <h3 className="mb-0">{j.title}</h3>
                <p className="mt-2" style={{whiteSpace:'pre-wrap'}}>{j.description}</p>
                <div className="mt-2"><span className="badge">Skills: {j.skills.join(', ') || '—'}</span></div>
                <div className="row mt-3">
                  <div><span className="badge">Lương: {j.salaryMin ?? '—'} - {j.salaryMax ?? '—'}</span></div>
                  <div><span className="badge">Địa điểm: {j.location || '—'}</span></div>
                  <div><span className="badge">Hình thức: {j.employmentType}</span></div>
                </div>
                <div className="mt-3"><span className="badge">Đăng lúc: {new Date(j.postedAt).toLocaleString()}</span></div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
