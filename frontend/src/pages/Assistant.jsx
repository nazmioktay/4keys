export default function Assistant() {
  return (
    <div className="page">
      <h1 className="page-title">AI Asistan</h1>
      <div className="card">
        <div className="card-title">Henüz eklenmedi</div>
        <p className="muted">
          Bu sekme, orijinal 3Commas tasarımındaki "AI Asistan" ile görsel
          bütünlüğü korumak için duruyor — ama arkasında henüz gerçek bir
          backend özelliği yok. Sahte bir sohbet kutusu göstermek yerine
          bunu dürüstçe belirtmeyi tercih ettik.
        </p>
        <p className="muted">
          İsterseniz burayı, sistemin mevcut modüllerini (screener, ML
          tahminleri, backtest sonuçları) doğal dilde yorumlayan gerçek bir
          Claude API entegrasyonuna bağlayabiliriz.
        </p>
      </div>
    </div>
  );
}
