# Enterprise RAG System

PDF dokümanlarını indeksleyen ve kaynak göstererek yanıtlayan production odaklı
RAG sistemi.

Sistem şu servislerden oluşur:

- FastAPI backend
- Streamlit kullanıcı arayüzü
- Ollama LLM ve embedding servisi
- ChromaDB vektör veritabanı
- RAG değerlendirme ve kalite kapısı

## Gereksinimler

- Docker Desktop veya Docker Engine
- Docker Compose v2
- En az 8 GB RAM
- Model dosyaları için en az 10 GB boş disk alanı

İlk kurulumda yaklaşık 6.1 GB büyüklüğündeki `llama3.1:8b` ve `bge-m3`
modelleri otomatik indirilir.

## Hızlı Başlangıç

### 1. Environment dosyasını oluştur

PowerShell:

```powershell
Copy-Item .env.production.example .env
```

Linux veya macOS:

```bash
cp .env.production.example .env
```

`.env` dosyasında `API_KEY` ve `RAG_API_KEY` alanlarını aynı güçlü ve rastgele
değerle değiştir.

PowerShell ile güvenli bir anahtar üretmek için:

```powershell
$bytes = New-Object byte[] 32
[System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
[Convert]::ToBase64String($bytes)
```

Üretilen değeri iki alana da yaz:

```env
API_KEY=GENERATED_SECRET
RAG_API_KEY=GENERATED_SECRET
```

`.env` dosyasını Git'e ekleme.

### 2. Servisleri başlat

```powershell
docker compose up --detach --build
```

İlk model indirme işlemini takip et:

```powershell
docker compose logs --follow ollama-models
```

Hazır olduğunda şu mesaj görüntülenir:

```text
Required Ollama models are ready.
```

### 3. Servisleri doğrula

```powershell
docker compose ps
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8501/_stcore/health
```

Beklenen servis durumları:

- `ollama`: healthy
- `ollama-models`: exited (0)
- `api`: healthy
- `ui`: healthy

`ollama-models` tek seferlik başlangıç servisidir. Modelleri indirdikten sonra
`Exited (0)` durumunda olması normaldir.

## Erişim Adresleri

- Streamlit UI: http://127.0.0.1:8501
- FastAPI dokümantasyonu: http://127.0.0.1:8000/docs
- API health check: http://127.0.0.1:8000/health
- Prometheus metrikleri: http://127.0.0.1:8000/metrics

Ollama portu güvenlik amacıyla host sistemine açılmaz; yalnızca Compose ağı
içindeki API servisi tarafından erişilebilir.

## İlk Kullanım

1. Streamlit arayüzünü aç.
2. Sidebar üzerinden bir PDF yükle.
3. İndeksleme tamamlandıktan sonra doküman hakkında soru sor.
4. Yanıtın altındaki kaynak, sayfa ve skor bilgilerini kontrol et.

## Yaygın Docker Komutları

```powershell
# Servis durumları
docker compose ps

# Tüm loglar
docker compose logs --follow

# Yalnızca API logları
docker compose logs --follow api

# Servisleri yeniden başlat
docker compose restart

# Servisleri durdur; verileri koru
docker compose down

# Image'ları yeniden oluştur
docker compose build api ui
```

Named volume'lar sayesinde `docker compose down` sonrasında Chroma verileri ve
Ollama modelleri korunur.

## Geliştirici Komutları

Windows:

```powershell
.\scripts\dev.ps1 help
.\scripts\dev.ps1 quality
.\scripts\dev.ps1 docker-config
```

GNU Make bulunan sistemler:

```bash
make help
make quality
make docker-config
```

Integration testleri çalışan Ollama ve Chroma servisleri gerektirir:

```powershell
.\scripts\dev.ps1 test-all
```

## Production Notları

Varsayılan olarak API ve UI yalnızca `127.0.0.1` üzerinde yayınlanır. Uzak
erişim için TLS kullanan bir reverse proxy üzerinden yayınlanmaları önerilir.

Production ortamında:

- `.env` dosyasını Git'e ekleme.
- Güçlü ve benzersiz bir API anahtarı kullan.
- `BIND_ADDRESS` değerini yalnızca ağ mimarisi gerektiriyorsa değiştir.
- Chroma ve Ollama volume'larını düzenli olarak yedekle.
