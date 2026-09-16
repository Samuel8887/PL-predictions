const labels = {'premier-league': 'Premier League', championship: 'Championship', 'league-one': 'League One'};
let data, colours = {}, league = 'premier-league', view = 'matches', gameweek = null;
const esc = value => String(value).replace(/[&<>"']/g, character => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character]));
const pct = value => `${Math.round(value * 100)}%`;
const colour = team => colours[team] || '#52657b';
const ordinal = value => `${value}${value % 100 >= 11 && value % 100 <= 13 ? 'th' : ({1:'st',2:'nd',3:'rd'}[value % 10] || 'th')}`;

function lineupRows(players) {
  return players.map(player => `<tr><td>${esc(player.player)}</td><td>${esc(player.position)}</td><td>${pct(player.start_probability)}</td><td>${player.minutes}</td><td>${player.goals}</td><td>${player.assists}</td><td>${Number(player.xg).toFixed(2)}</td><td>${Number(player.xa).toFixed(2)}</td><td>${pct(player.score_probability)}</td><td>${pct(player.assist_probability)}</td></tr>`).join('');
}
function squadTable(players) {
  return `<div class="table-wrap"><table class="lineup-table"><thead><tr>${['Player','Pos.','Start','Min.','G','A','xG','xA','Score','Assist'].map(label => `<th scope="col">${label}</th>`).join('')}</tr></thead><tbody>${lineupRows(players)}</tbody></table></div>`;
}
function playerPicks(sides) {
  if (!sides?.length) return '';
  return `<section class="match-details"><h2>Projected squads & player statistics</h2><p>The XI selects eligible players in a plausible formation; up to nine others form the projected bench. It is not a confirmed team sheet. Start and performance chances are heuristic estimates. Min., G, A, xG and xA are current-season totals.</p>${sides.map(side => `<section class="player-team"><h3>${esc(side.team)} · projected XI</h3><p><strong>Scorer pick:</strong> ${esc(side.top_scorer.player)} (${pct(side.top_scorer.score_probability)}) · <strong>Assist pick:</strong> ${esc(side.top_assister.player)} (${pct(side.top_assister.assist_probability)})</p>${side.projected_lineup.length ? squadTable(side.projected_lineup) : '<p>Not enough eligible players for a complete formation.</p>'}<h3 class="bench-title">Projected bench</h3>${squadTable(side.projected_bench)}</section>`).join('')}</section>`;
}
function card(fixture) {
  const date = new Date(fixture.date + 'T12:00').toLocaleDateString(undefined, {weekday:'short',month:'short',day:'numeric',year:'numeric'});
  const heading = `<div class="fixture-head">${esc(date)}${fixture.kickoff ? ' · ' + esc(fixture.kickoff) : ''}<span>Match details ↗</span></div><div class="teams"><span class="accent" style="--team:${colour(fixture.home_team)}">${esc(fixture.home_team)}</span><span>VS</span><span class="away"><span class="accent" style="--team:${colour(fixture.away_team)}">${esc(fixture.away_team)}</span></span></div>`;
  if (!fixture.prediction_available) return `<details class="fixture"><summary>${heading}<div class="summary">No statistical estimate yet — insufficient prior data.</div></summary></details>`;
  const probabilities = [fixture.home_win, fixture.draw, fixture.away_win];
  const outcomes = [`${fixture.home_team} win`, 'Draw', `${fixture.away_team} win`];
  const winner = outcomes[probabilities.indexOf(Math.max(...probabilities))];
  return `<details class="fixture"><summary>${heading}<div class="prob-labels"><span>${pct(probabilities[0])} home</span><span>${pct(probabilities[1])} draw</span><span>${pct(probabilities[2])} away</span></div><div class="bar" aria-hidden="true"><i class="home" style="width:${probabilities[0]*100}%"></i><i class="draw" style="width:${probabilities[1]*100}%"></i><i class="awaybar" style="width:${probabilities[2]*100}%"></i></div><div class="summary">Leading outcome: <strong>${esc(winner)}</strong> · Expected goals ${fixture.expected_home_goals.toFixed(1)} – ${fixture.expected_away_goals.toFixed(1)}${fixture.context === 'cross-division' ? ' · Limited division history' : ''}</div></summary>${playerPicks(fixture.player_predictions)}</details>`;
}
function table(outlook) {
  const showPoints = outlook.table.every(row => Number.isFinite(row.expected_points));
  return `<div class="outlook"><h2>Season outlook</h2><p>${esc(outlook.season)} · ${outlook.simulations.toLocaleString()} simulations. Expected finish averages every simulation; most likely finish is the most frequent individual position.</p><div class="table-wrap"><table class="league-table"><thead><tr><th scope="col">Expected finish</th><th scope="col">Team</th>${showPoints ? '<th scope="col">Expected points</th>' : ''}<th scope="col">Most likely finish</th><th scope="col">Title chance</th></tr></thead><tbody>${outlook.table.map(row => `<tr><td class="position">${row.expected_position.toFixed(1)}</td><td>${esc(row.team)}</td>${showPoints ? `<td>${row.expected_points.toFixed(1)}</td>` : ''}<td>${ordinal(row.most_likely_position)} · ${pct(row.most_likely_position_probability)}</td><td>${pct(row.win_probability)}</td></tr>`).join('')}</tbody></table></div><div class="explanation"><h2>How to read this table</h2><p>Remaining fixtures are sampled from score distributions calibrated to the match outcome probabilities. Simulated results are added to existing points and goals. Team strengths are held fixed. Exact ties after points, goal difference and goals scored share positions equally; head-to-head and playoffs are not modelled. Displayed percentages are rounded.</p></div></div>`;
}
const $ = selector => document.querySelector(selector);
let search = '';
const weeksForLeague = () => [...new Set((data.leagues[league] || []).map(f => f.matchweek).filter(Number.isInteger))].sort((a, b) => a - b);

function render() {
  const fixtures = data.leagues[league] || [];
  const isPremier = league === 'premier-league';
  const showTable = isPremier && view === 'table';
  const weeks = weeksForLeague();
  if (!weeks.includes(gameweek)) gameweek = weeks[0] ?? null;
  const weekIndex = weeks.indexOf(gameweek);
  const selected = fixtures.filter(f => (search || gameweek === null || f.matchweek === gameweek) && (!search || `${f.home_team} ${f.away_team}`.toLowerCase().includes(search)));
  $('#premier-views').hidden = !isPremier;
  $('#gameweek-nav').hidden = showTable || gameweek === null || Boolean(search);
  $('#team-search').disabled = showTable;
  $('#gameweek-title').textContent = `Gameweek ${gameweek ?? '—'}`;
  $('#previous-gameweek').disabled = weekIndex <= 0;
  $('#next-gameweek').disabled = weekIndex < 0 || weekIndex >= weeks.length - 1;
  $('#metric-league').textContent = labels[league];
  $('#metric-count').textContent = fixtures.length;
  const estimated = fixtures.filter(f => f.prediction_available);
  $('#metric-goals').textContent = estimated.length ? (estimated.reduce((sum, f) => sum + f.expected_home_goals + f.expected_away_goals, 0) / estimated.length).toFixed(2) : '—';
  $('#status').textContent = showTable ? `${labels[league]} · season outlook` : `${selected.length} fixtures · ${selected.filter(f => f.prediction_available).length} estimates${search ? ' · all gameweeks' : ''}`;
  $('#fixtures').hidden = showTable;
  // Only create squad tables when a user opens a card.
  $('#fixtures').innerHTML = showTable ? '' : selected.length ? selected.map((fixture, index) => card({...fixture, player_predictions: null}).replace('<details ', `<details data-index="${index}" `)).join('') : '<p class="empty">No fixtures found. Try another club or competition.</p>';
  $('#fixtures').querySelectorAll('details').forEach(detail => {
    detail.addEventListener('toggle', () => {
      if (detail.open && !detail.dataset.loaded) {
        detail.insertAdjacentHTML('beforeend', playerPicks(selected[Number(detail.dataset.index)].player_predictions));
        detail.dataset.loaded = 'true';
      }
    });
  });
  const outlook = data.season_outlook?.[league];
  $('#outlook').innerHTML = showTable ? (outlook?.table ? table(outlook) : '<p class="empty">Season outlook is unavailable. Rebuild forecasts with the updated model to generate it.</p>') : '';
  document.querySelectorAll('button[data-league]').forEach(button => {
    button.classList.toggle('active', button.dataset.league === league);
    button.setAttribute('aria-pressed', String(button.dataset.league === league));
  });
  document.querySelectorAll('button[data-view]').forEach(button => {
    button.classList.toggle('active', button.dataset.view === view);
    button.setAttribute('aria-pressed', String(button.dataset.view === view));
  });
}

document.querySelectorAll('button[data-league]').forEach(button => button.addEventListener('click', () => {
  league = button.dataset.league; view = 'matches'; gameweek = null; render();
}));
document.querySelectorAll('button[data-view]').forEach(button => button.addEventListener('click', () => {
  view = button.dataset.view; render();
}));
$('#team-search').addEventListener('input', event => { search = event.target.value.trim().toLowerCase(); render(); });
for (const [selector, delta] of [['#previous-gameweek', -1], ['#next-gameweek', 1]]) {
  $(selector).addEventListener('click', () => {
    const weeks = weeksForLeague(), next = weeks.indexOf(gameweek) + delta;
    if (next >= 0 && next < weeks.length) { gameweek = weeks[next]; render(); }
  });
}
async function fetchJSON(path) {
  const response = await fetch(path, {cache: 'no-cache'});
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}
async function load() {
  try {
    const [predictions, palette, report] = await Promise.all([
      fetchJSON('data/predictions.json'),
      fetchJSON('data/team_colours.json').catch(() => ({})),
      fetchJSON('data/evaluation.json').catch(() => null)
    ]);
    if (!predictions.leagues || !predictions.generated_at) throw new Error('Invalid forecast data');
    data = predictions;
    colours = Object.fromEntries(Object.entries(palette).filter(([, value]) => /^#[0-9a-f]{6}$/i.test(value)));
    const generated = new Date(data.generated_at);
    $('#updated').textContent = `Forecast generated ${generated.toLocaleString()}.`;
    $('#metric-date').textContent = generated.toLocaleDateString(undefined, {day: 'numeric', month: 'short'});
    const legacy = data.snapshot_kind === 'legacy';
    const stale = Date.now() - generated.getTime() > 48 * 60 * 60 * 1000;
    $('#data-notice').hidden = !legacy && !stale;
    $('#data-notice').textContent = legacy ? 'Archived forecast preview · These match estimates came with the original project. Rebuild to apply the updated model. Previous squad and season projections have been withheld because they were affected by calculation errors.' : 'This forecast is over 48 hours old. Results, schedules and availability may have changed.';
    if (report?.fixtures) $('#evaluation').textContent = `${legacy ? 'Original model evaluation' : 'Chronological evaluation'}: ${report.fixtures} fixtures · log loss ${report.log_loss} (baseline ${report.baseline_log_loss}) · accuracy ${pct(report.accuracy)}. Lower log loss is better. Player estimates are not validated by this report.`;
    render();
  } catch (error) {
    $('#status').innerHTML = 'Forecasts could not be loaded. <button id="retry">Try again</button>';
    $('#retry').addEventListener('click', load);
  }
}
load();
