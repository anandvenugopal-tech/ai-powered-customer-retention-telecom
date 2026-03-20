import './App.css';
import { useMemo, useState } from 'react';

function App() {
  const [customerId, setCustomerId] = useState('');
  const [customer, setCustomer] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [status, setStatus] = useState({ state: 'idle', message: '' }); // idle | loading | error | success

  const apiBaseUrl =
    (typeof process !== 'undefined' && process.env && process.env.REACT_APP_API_BASE_URL) || '';

  const canFetch = customerId.trim().length > 0 && status.state !== 'loading';

  const churnTone = useMemo(() => {
    if (!prediction) return 'neutral';
    const label = String(prediction.label ?? prediction.churn ?? prediction.result ?? '').toLowerCase();
    const prob = prediction.probability ?? prediction.churn_probability ?? prediction.score;

    if (label.includes('yes') || label.includes('churn') || label.includes('true')) return 'danger';
    if (label.includes('no') || label.includes('stay') || label.includes('false')) return 'success';
    if (typeof prob === 'number') return prob >= 0.5 ? 'danger' : 'success';
    return 'neutral';
  }, [prediction]);

  const customerRows = useMemo(() => {
    if (!customer || typeof customer !== 'object') return [];

    const exclude = new Set(['prediction', 'churn', 'result']);

    const preferred = [
      { key: 'customerID', label: 'Customer ID' },
      { key: 'customer_id', label: 'Customer ID' },
      { key: 'tenure', label: 'Tenure (months)' },
      { key: 'Contract', label: 'Contract' },
      { key: 'MonthlyCharges', label: 'Monthly Charges' },
      { key: 'TotalCharges', label: 'Total Charges' },
      { key: 'SeniorCitizen', label: 'Senior Citizen' },
      { key: 'Partner', label: 'Partner' },
      { key: 'Dependents', label: 'Dependents' },
      { key: 'InternetService', label: 'Internet Service' },
      { key: 'PhoneService', label: 'Phone Service' },
      { key: 'PaymentMethod', label: 'Payment Method' },
    ];

    const presentPreferred = preferred
      .filter((f) => Object.prototype.hasOwnProperty.call(customer, f.key))
      .map((f) => ({ key: f.key, label: f.label, value: customer[f.key] }));

    const used = new Set(presentPreferred.map((r) => r.key));
    const rest = Object.entries(customer)
      .filter(([k]) => !exclude.has(k) && !used.has(k))
      .slice(0, Math.max(0, 18 - presentPreferred.length))
      .map(([key, value]) => ({ key, label: prettifyKey(key), value }));

    return [...presentPreferred, ...rest];
  }, [customer]);

  function prettifyKey(key) {
    return String(key)
      .replace(/[_-]+/g, ' ')
      .replace(/([a-z])([A-Z])/g, '$1 $2')
      .replace(/\s+/g, ' ')
      .trim()
      .replace(/^./, (c) => c.toUpperCase());
  }

  function formatValue(value) {
    if (value === null || value === undefined) return '—';
    if (typeof value === 'number') return String(value);
    if (typeof value === 'boolean') return value ? 'Yes' : 'No';
    if (typeof value === 'string') return value.trim() ? value : '—';
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  }

  async function fetchCustomer() {
    const id = customerId.trim();
    if (!id) return;

    setStatus({ state: 'loading', message: '' });
    setCustomer(null);
    setPrediction(null);

    try {
      // Expected response shape (flexible):
      // { customer: {...}, prediction: { label, probability } }
      // OR { ...customerFields, prediction: {...} }
      const res = await fetch(`http://127.0.0.1:8000/customer/${id}`);

      if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw new Error(text || `Request failed (${res.status})`);
      }

      const data = await res.json();
      const customerPayload = data.customer ?? data.data ?? data;

      // Keep existing customer display logic unchanged.
      setCustomer(customerPayload && typeof customerPayload === 'object' ? customerPayload : null);

      // Clean data for prediction.
      const inputData = { ...(customerPayload && typeof customerPayload === 'object' ? customerPayload : {}) };
      delete inputData.customer_id;
      delete inputData.churn;

      try {
        const predictRes = await fetch('http://127.0.0.1:8000/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(inputData),
        });

        if (!predictRes.ok) {
          const text = await predictRes.text().catch(() => '');
          throw new Error(text || `Prediction request failed (${predictRes.status})`);
        }

        // const predictionResponse = await predictRes.json().catch(() => null);
        // const predictionPayload =
        //   predictionResponse?.prediction ?? predictionResponse?.churn ?? predictionResponse?.result ?? predictionResponse ?? null;
        let predictionPayload = null;

        if (data.prediction) {
          predictionPayload = data.prediction;
        }else if (data.churn !== undefined) {
          predictionPayload = {
            label: data.churn === 1 ? "Churn" : "No Churn",
            probability: data.probabilty ?? null
          };
        }

        setPrediction(predictionPayload && typeof predictionPayload === 'object' ? predictionPayload : predictionPayload);
        setStatus({ state: 'success', message: 'Customer data loaded.' });
      } catch (e) {
        setPrediction(null);
        setStatus({
          state: 'error',
          message: e?.message ? `Prediction failed: ${String(e.message)}` : 'Prediction failed.',
        });
      }
    } catch (e) {
      setStatus({
        state: 'error',
        message: e?.message ? String(e.message) : 'Failed to fetch customer data.',
      });
    }
  }

  return (
    <div className="Page">
      <main className="Shell">
        <header className="Header">
          <div className="TitleRow">
            <div className="Dot Dot--blue" aria-hidden="true" />
            <h1 className="Title">Customer Retention Dashboard</h1>
          </div>
          <p className="Subtitle">Look up a customer and view churn prediction in a clean, minimal layout.</p>
        </header>

        <section className="Controls" aria-label="Customer lookup">
          <label className="Label" htmlFor="customerId">
            Customer ID
          </label>
          <div className="ControlRow">
            <input
              id="customerId"
              className="Input"
              value={customerId}
              onChange={(e) => setCustomerId(e.target.value)}
              placeholder="e.g., 0001-AFZ"
              inputMode="text"
              autoComplete="off"
            />
            <button className="Button" type="button" onClick={fetchCustomer} disabled={!canFetch}>
              {status.state === 'loading' ? 'Fetching…' : 'Fetch customer'}
            </button>
          </div>

          {status.state === 'error' ? (
            <div className="Callout Callout--danger" role="alert">
              <span className="CalloutDot" aria-hidden="true" />
              <span>{status.message || 'Could not fetch customer.'}</span>
            </div>
          ) : status.state === 'success' ? (
            <div className="Callout Callout--success" role="status">
              <span className="CalloutDot" aria-hidden="true" />
              <span>{status.message}</span>
            </div>
          ) : (
            <div className="Callout Callout--neutral" role="note">
              <span className="CalloutDot" aria-hidden="true" />
              <span>
                API endpoint: <code className="InlineCode">GET {apiBaseUrl || ''}/api/customers/:id</code>
              </span>
            </div>
          )}
        </section>

        <section className="Grid" aria-label="Customer details and churn prediction">
          <article className="Card">
            <div className="CardHeader">
              <h2 className="CardTitle">Customer Info</h2>
              <span className="Badge Badge--info">Blue</span>
            </div>

            {!customer ? (
              <div className="Empty">
                <p className="EmptyTitle">No customer loaded</p>
                <p className="EmptyText">Enter a customer ID above and fetch to see details.</p>
              </div>
            ) : (
              <dl className="KeyValue">
                {customerRows.map((row) => (
                  <div key={row.key} className="KeyValueRow">
                    <dt className="Key">{row.label}</dt>
                    <dd className="Value">{formatValue(row.value)}</dd>
                  </div>
                ))}
              </dl>
            )}
          </article>

          <article className="Card">
            <div className="CardHeader">
              <h2 className="CardTitle">Churn Prediction</h2>
              <span
                className={[
                  'Badge',
                  churnTone === 'danger'
                    ? 'Badge--danger'
                    : churnTone === 'success'
                      ? 'Badge--success'
                      : 'Badge--neutral',
                ].join(' ')}
              >
                {churnTone === 'danger' ? 'Risk' : churnTone === 'success' ? 'Healthy' : 'Neutral'}
              </span>
            </div>

            {!prediction ? (
              <div className="Empty">
                <p className="EmptyTitle">No prediction yet</p>
                <p className="EmptyText">Fetch a customer to see the churn result.</p>
              </div>
            ) : typeof prediction === 'object' ? (
              <div className="Prediction">
                <div className="MetricRow">
                  <div className="Metric">
                    <div className="MetricLabel">Label</div>
                    <div className="MetricValue">{String(prediction.label ?? prediction.churn ?? prediction.result ?? '—')}</div>
                  </div>
                  <div className="Metric">
                    <div className="MetricLabel">Probability</div>
                    <div className="MetricValue">
                      {typeof (prediction.probability ?? prediction.churn_probability ?? prediction.score) === 'number'
                        ? `${Math.round(
                            100 * (prediction.probability ?? prediction.churn_probability ?? prediction.score),
                          )}%`
                        : String(prediction.probability ?? prediction.churn_probability ?? prediction.score ?? '—')}
                    </div>
                  </div>
                </div>

                <div
                  className={[
                    'SoftPanel',
                    churnTone === 'danger'
                      ? 'SoftPanel--danger'
                      : churnTone === 'success'
                        ? 'SoftPanel--success'
                        : 'SoftPanel--neutral',
                  ].join(' ')}
                >
                  <div className="SoftPanelTitle">Recommendation</div>
                  <div className="SoftPanelText">
                    {churnTone === 'danger'
                      ? 'High churn risk. Consider proactive outreach, plan review, and service quality checks.'
                      : churnTone === 'success'
                        ? 'Low churn risk. Maintain satisfaction and monitor key usage signals.'
                        : 'Review the customer profile for signals and confirm prediction confidence.'}
                  </div>
                </div>
              </div>
            ) : (
              <div className="Prediction">
                <div className="SoftPanel SoftPanel--neutral">
                  <div className="SoftPanelTitle">Result</div>
                  <div className="SoftPanelText">{String(prediction)}</div>
                </div>
              </div>
            )}
          </article>
        </section>

        <footer className="Footer">
          <span className="FooterText">Light theme • Minimal layout • Soft colors</span>
        </footer>
      </main>
    </div>
  );
}

export default App;
